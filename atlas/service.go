package atlas

import (
	"context"
	"crypto/subtle"
	"database/sql"
	"errors"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/qdrant/go-client/qdrant"
	"github.com/riverqueue/river"
	"github.com/riverqueue/river/riverdriver/riverdatabasesql"
	"github.com/riverqueue/river/rivermigrate"
)

func LoadConfig() (Config, error) {
	cfg := Config{Addr: env("ATLAS_ADDR", ":8090"), DatabaseURL: os.Getenv("DATABASE_URL"), ServiceToken: os.Getenv("ATLAS_SERVICE_TOKEN")}
	if cfg.DatabaseURL == "" || cfg.ServiceToken == "" {
		return Config{}, errors.New("DATABASE_URL and ATLAS_SERVICE_TOKEN are required")
	}
	return cfg, nil
}

func NewService(cfg Config, db *sql.DB) (*Service, error) {
	dimension, err := intEnv("TEI_EMBEDDING_DIMENSION", 384)
	if err != nil {
		return nil, err
	}
	port, err := intEnv("QDRANT_GRPC_PORT", 6334)
	if err != nil {
		return nil, err
	}
	attempts, err := intEnv("ATLAS_INGESTION_ATTEMPTS", 3)
	if err != nil {
		return nil, err
	}
	uploadLimit, err := intEnv("ATLAS_MAX_UPLOAD_BYTES", 104857600)
	if err != nil {
		return nil, err
	}
	denseThreshold, err := floatEnv("ATLAS_DENSE_THRESHOLD")
	if err != nil {
		return nil, err
	}
	sparseThreshold, err := floatEnv("ATLAS_SPARSE_THRESHOLD")
	if err != nil {
		return nil, err
	}
	useTLS, err := boolEnv("QDRANT_USE_TLS", false)
	if err != nil {
		return nil, err
	}
	store, err := newObjectStore()
	if err != nil {
		return nil, err
	}
	client, err := qdrant.NewClient(&qdrant.Config{Host: env("QDRANT_HOST", "localhost"), Port: port, APIKey: os.Getenv("QDRANT_API_KEY"), UseTLS: useTLS})
	if err != nil {
		return nil, fmt.Errorf("connect qdrant: %w", err)
	}

	s := &Service{
		db: db, store: store, serviceToken: cfg.ServiceToken,
		indexVersion: env("ATLAS_INDEX_VERSION", "v1"), queryPrefix: env("TEI_QUERY_PREFIX", "Represent this sentence for searching relevant passages: "),
		embeddingDimension: dimension, maxAttempts: attempts, maxUploadBytes: int64(uploadLimit),
		denseThreshold: denseThreshold, sparseThreshold: sparseThreshold,
	}
	modelHTTP := &http.Client{Timeout: 30 * time.Minute}
	s.processor = &processorClient{modelClient{url: env("ATLAS_PROCESSOR_URL", "http://localhost:5001"), name: "processor", http: modelHTTP}}
	s.embedder = &teiClient{modelClient{url: env("TEI_URL", "http://localhost:8081"), name: "TEI", http: modelHTTP}}
	s.index = &qdrantIndex{client: client, denseCollection: env("QDRANT_ROUTE_COLLECTION", "atlas_routes_v1"), sparseCollection: env("QDRANT_CHUNK_COLLECTION", "atlas_chunks_v1"), embeddingDimension: dimension}
	workers := river.NewWorkers()
	river.AddWorker(workers, &ingestionWorker{service: s})
	s.jobs, err = river.NewClient(riverdatabasesql.New(db), &river.Config{JobTimeout: 30 * time.Minute, Queues: map[string]river.QueueConfig{river.QueueDefault: {MaxWorkers: 1}}, Workers: workers, ErrorHandler: &ingestionErrorHandler{db: db}})
	if err != nil {
		client.Close()
		return nil, fmt.Errorf("configure River: %w", err)
	}
	return s, nil
}

func (s *Service) PrepareInfrastructure(ctx context.Context) error {
	migrator, err := rivermigrate.New(riverdatabasesql.New(s.db), nil)
	if err != nil {
		return err
	}
	if _, err = migrator.Migrate(ctx, rivermigrate.DirectionUp, nil); err != nil {
		return fmt.Errorf("migrate River: %w", err)
	}
	return s.index.PrepareCollections(ctx)
}

func (s *Service) RunWorker(ctx context.Context) error {
	if err := s.jobs.Start(ctx); err != nil {
		return err
	}
	<-ctx.Done()
	stop, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	return s.jobs.Stop(stop)
}

func (s *Service) Close() {
	if index, ok := s.index.(*qdrantIndex); ok {
		index.client.Close()
	}
}

func (s *Service) Router() http.Handler {
	router := gin.New()
	router.Use(gin.Recovery())
	router.GET("/health", func(c *gin.Context) { c.JSON(http.StatusOK, gin.H{"status": "ok"}) })
	api := router.Group("/v1", func(c *gin.Context) {
		if subtle.ConstantTimeCompare([]byte(c.GetHeader("Authorization")), []byte("Bearer "+s.serviceToken)) != 1 {
			c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{"error": "unauthorized"})
			return
		}
		namespaceID := c.GetHeader("X-Atlas-Namespace-ID")
		if _, err := uuid.Parse(namespaceID); err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "invalid namespace"})
			return
		}
		c.Set("namespaceID", namespaceID)
	})
	api.GET("/collections", s.listCollections)
	api.POST("/collections", s.createCollection)
	api.GET("/collections/:collection_id/documents", s.listDocuments)
	api.POST("/collections/:collection_id/documents", s.uploadDocument)
	api.DELETE("/collections/:collection_id/documents/:document_id", s.deleteDocument)
	api.POST("/retrieve", s.handleRetrieve)
	return router
}

func (s *Service) listCollections(c *gin.Context) {
	rows, err := s.db.QueryContext(c, `SELECT id::text,name,description,index_version,created_at FROM atlas_collections WHERE namespace_id=$1::uuid ORDER BY created_at DESC`, c.GetString("namespaceID"))
	if err != nil {
		s.writeError(c, err)
		return
	}
	defer rows.Close()
	items := []Collection{}
	for rows.Next() {
		var item Collection
		if err := rows.Scan(&item.ID, &item.Name, &item.Description, &item.IndexVersion, &item.CreatedAt); err != nil {
			s.writeError(c, err)
			return
		}
		items = append(items, item)
	}
	if rows.Err() != nil {
		s.writeError(c, rows.Err())
		return
	}
	c.JSON(http.StatusOK, gin.H{"data": items})
}

func (s *Service) createCollection(c *gin.Context) {
	var input struct{ Name, Description string }
	if c.ShouldBindJSON(&input) != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid request"})
		return
	}
	input.Name, input.Description = strings.TrimSpace(input.Name), strings.TrimSpace(input.Description)
	if input.Name == "" || utf8.RuneCountInString(input.Name) > 100 || utf8.RuneCountInString(input.Description) > 1000 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "name must be 1-100 characters and description at most 1,000 characters"})
		return
	}
	var item Collection
	err := s.db.QueryRowContext(c, `INSERT INTO atlas_collections(namespace_id,name,description,index_version) VALUES($1::uuid,$2,$3,$4) RETURNING id::text,name,description,index_version,created_at`, c.GetString("namespaceID"), input.Name, input.Description, s.indexVersion).Scan(&item.ID, &item.Name, &item.Description, &item.IndexVersion, &item.CreatedAt)
	if err != nil {
		s.writeError(c, err)
		return
	}
	c.JSON(http.StatusCreated, gin.H{"data": item})
}

func (s *Service) listDocuments(c *gin.Context) {
	namespaceID, collectionID := c.GetString("namespaceID"), c.Param("collection_id")
	if _, err := uuid.Parse(collectionID); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid collection_id"})
		return
	}
	limit, offset := 20, 0
	var err error
	if c.Query("limit") != "" {
		limit, err = strconv.Atoi(c.Query("limit"))
	}
	if c.Query("offset") != "" {
		offset, err = strconv.Atoi(c.Query("offset"))
	}
	if err != nil || limit < 1 || limit > 50 || offset < 0 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "limit must be 1-50 and offset must be positive"})
		return
	}
	rows, err := s.db.QueryContext(c, `SELECT d.id::text,d.collection_id::text,d.filename,d.mime_type,d.size_bytes,d.sha256,d.status,d.index_stage,d.failure_reason,d.route_count,d.chunk_count,d.created_at FROM atlas_documents d JOIN atlas_collections b ON b.id=d.collection_id WHERE b.namespace_id=$1::uuid AND b.id=$2::uuid ORDER BY d.created_at DESC LIMIT $3 OFFSET $4`, namespaceID, collectionID, limit+1, offset)
	if err != nil {
		s.writeError(c, err)
		return
	}
	defer rows.Close()
	items := []Document{}
	for rows.Next() {
		var item Document
		if err := rows.Scan(&item.ID, &item.CollectionID, &item.Filename, &item.MimeType, &item.SizeBytes, &item.SHA256, &item.Status, &item.IndexStage, &item.FailureReason, &item.RouteCount, &item.ChunkCount, &item.CreatedAt); err != nil {
			s.writeError(c, err)
			return
		}
		items = append(items, item)
	}
	if rows.Err() != nil {
		s.writeError(c, rows.Err())
		return
	}
	var nextOffset any
	if len(items) > limit {
		items, nextOffset = items[:limit], offset+limit
	}
	c.JSON(http.StatusOK, gin.H{"data": items, "next_offset": nextOffset})
}

func (s *Service) uploadDocument(c *gin.Context) {
	namespaceID, collectionID := c.GetString("namespaceID"), c.Param("collection_id")
	if _, err := uuid.Parse(collectionID); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid collection_id"})
		return
	}
	c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, s.maxUploadBytes+(1<<20))
	header, err := c.FormFile("file")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "a document file is required"})
		return
	}
	name := filepath.Base(strings.TrimSpace(header.Filename))
	switch strings.ToLower(filepath.Ext(name)) {
	case ".pdf", ".md", ".markdown", ".txt":
	default:
		c.JSON(http.StatusBadRequest, gin.H{"error": "supported document types are PDF, Markdown, and plain text"})
		return
	}
	file, err := header.Open()
	if err != nil {
		s.writeError(c, err)
		return
	}
	defer file.Close()
	documentID := uuid.NewString()
	storageKey := strings.Join([]string{namespaceID, collectionID, documentID, name}, "/")
	// Store the source first, then commit its metadata and dense-stage River job together.
	stored, err := s.store.Put(c, storageKey, file, s.maxUploadBytes)
	if err != nil {
		s.writeError(c, err)
		return
	}
	cleanup := func(cause error) error {
		if removeErr := s.store.Delete(c, storageKey); removeErr != nil {
			return errors.Join(cause, fmt.Errorf("remove upload: %w", removeErr))
		}
		return cause
	}
	tx, err := s.db.BeginTx(c, nil)
	if err != nil {
		s.writeError(c, cleanup(err))
		return
	}
	defer tx.Rollback()
	var item Document
	err = tx.QueryRowContext(c, `WITH base AS (SELECT id,index_generation FROM atlas_collections WHERE id=$2::uuid AND namespace_id=$8::uuid FOR UPDATE) INSERT INTO atlas_documents(id,collection_id,filename,mime_type,size_bytes,sha256,storage_key,status,index_generation) SELECT $1::uuid,base.id,$3,$4,$5,$6,$7,'queued',base.index_generation FROM base ON CONFLICT(collection_id,sha256) DO NOTHING RETURNING id::text,collection_id::text,filename,mime_type,size_bytes,sha256,status,index_stage,failure_reason,route_count,chunk_count,created_at`, documentID, collectionID, name, header.Header.Get("Content-Type"), stored.Size, stored.SHA256, storageKey, namespaceID).Scan(&item.ID, &item.CollectionID, &item.Filename, &item.MimeType, &item.SizeBytes, &item.SHA256, &item.Status, &item.IndexStage, &item.FailureReason, &item.RouteCount, &item.ChunkCount, &item.CreatedAt)
	if errors.Is(err, sql.ErrNoRows) {
		err = tx.QueryRowContext(c, `SELECT d.id::text,d.collection_id::text,d.filename,d.mime_type,d.size_bytes,d.sha256,d.status,d.index_stage,d.failure_reason,d.route_count,d.chunk_count,d.created_at FROM atlas_documents d JOIN atlas_collections b ON b.id=d.collection_id WHERE b.namespace_id=$1::uuid AND d.collection_id=$2::uuid AND d.sha256=$3`, namespaceID, collectionID, stored.SHA256).Scan(&item.ID, &item.CollectionID, &item.Filename, &item.MimeType, &item.SizeBytes, &item.SHA256, &item.Status, &item.IndexStage, &item.FailureReason, &item.RouteCount, &item.ChunkCount, &item.CreatedAt)
		if err == nil {
			err = tx.Commit()
		}
		if err != nil {
			s.writeError(c, cleanup(err))
			return
		}
		if err := cleanup(nil); err != nil {
			s.writeError(c, err)
			return
		}
		c.JSON(http.StatusOK, gin.H{"data": item, "duplicate": true})
		return
	}
	if err == nil {
		_, err = s.jobs.InsertTx(c, tx, ingestionArgs{DocumentID: documentID, Stage: "dense"}, &river.InsertOpts{MaxAttempts: s.maxAttempts, UniqueOpts: river.UniqueOpts{ByArgs: true}})
	}
	if err == nil {
		err = tx.Commit()
	}
	if err != nil {
		s.writeError(c, cleanup(err))
		return
	}
	c.JSON(http.StatusAccepted, gin.H{"data": item, "duplicate": false})
}

func (s *Service) deleteDocument(c *gin.Context) {
	namespaceID, collectionID, documentID := c.GetString("namespaceID"), c.Param("collection_id"), c.Param("document_id")
	if _, err := uuid.Parse(collectionID); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid collection_id"})
		return
	}
	if _, err := uuid.Parse(documentID); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid document_id"})
		return
	}
	// Hide first so retrieval rejects the document while its files are removed.
	var storageKey string
	err := s.db.QueryRowContext(c, `UPDATE atlas_documents d SET status='failed',failure_reason='deletion cleanup pending',updated_at=now() FROM atlas_collections b WHERE d.id=$1::uuid AND d.collection_id=$2::uuid AND b.id=d.collection_id AND b.namespace_id=$3::uuid RETURNING d.storage_key`, documentID, collectionID, namespaceID).Scan(&storageKey)
	if err == nil {
		err = errors.Join(s.index.DeleteDocument(c, namespaceID, collectionID, documentID), s.store.Delete(c, storageKey), s.store.Delete(c, chunkManifestKey(storageKey)))
	}
	if err == nil {
		_, err = s.db.ExecContext(c, `DELETE FROM atlas_documents WHERE id=$1::uuid AND collection_id=$2::uuid`, documentID, collectionID)
	}
	if err != nil {
		s.writeError(c, err)
		return
	}
	c.Status(http.StatusNoContent)
}

func (s *Service) handleRetrieve(c *gin.Context) {
	var input struct {
		CollectionID string `json:"collection_id"`
		Query        string `json:"query"`
	}
	if c.ShouldBindJSON(&input) != nil || strings.TrimSpace(input.Query) == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "collection_id and query are required"})
		return
	}
	if _, err := uuid.Parse(input.CollectionID); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid collection_id"})
		return
	}
	items, err := s.retrieveRelevantChunks(c, c.GetString("namespaceID"), input.CollectionID, input.Query)
	if err != nil {
		s.writeError(c, err)
		return
	}
	c.JSON(http.StatusOK, gin.H{"data": items})
}

func (e clientError) Error() string { return e.message }

func (s *Service) writeError(c *gin.Context, err error) {
	var requestErr clientError
	switch {
	case errors.As(err, &requestErr):
		c.JSON(requestErr.status, gin.H{"error": requestErr.message})
	case errors.Is(err, sql.ErrNoRows):
		c.JSON(http.StatusNotFound, gin.H{"error": "not found"})
	default:
		log.Printf("atlas request failed: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "atlas service operation failed"})
	}
}

func env(name, fallback string) string {
	if value := os.Getenv(name); value != "" {
		return value
	}
	return fallback
}

func intEnv(name string, fallback int) (int, error) {
	value, err := strconv.Atoi(env(name, strconv.Itoa(fallback)))
	if err != nil || value <= 0 {
		return 0, fmt.Errorf("%s must be a positive integer", name)
	}
	return value, nil
}

func floatEnv(name string) (float32, error) {
	value, err := strconv.ParseFloat(os.Getenv(name), 32)
	if err != nil || value <= 0 {
		return 0, fmt.Errorf("%s must be a positive number", name)
	}
	return float32(value), nil
}

func boolEnv(name string, fallback bool) (bool, error) {
	return strconv.ParseBool(env(name, strconv.FormatBool(fallback)))
}
