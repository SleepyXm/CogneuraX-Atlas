package atlas

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"mime/multipart"
	"net/http"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/lib/pq"
	"github.com/qdrant/go-client/qdrant"
	"github.com/riverqueue/river"
	"github.com/riverqueue/river/rivertype"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

const qdrantBatchSize, denseRoutesPerDocument = 128, 32

func (ingestionArgs) Kind() string { return "atlas_document_ingestion" }

func (w *ingestionWorker) Work(ctx context.Context, job *river.Job[ingestionArgs]) error {
	stage := job.Args.Stage
	if stage == "" {
		stage = "dense"
	}
	started := time.Now()
	var err error
	switch stage {
	case "dense":
		err = w.runDenseStage(ctx, job.Args.DocumentID)
	case "sparse":
		err = w.runSparseStage(ctx, job.Args.DocumentID)
	default:
		err = fmt.Errorf("unknown ingestion stage %q", stage)
	}
	log.Printf("atlas ingestion stage=%s document=%s duration=%s success=%t", stage, job.Args.DocumentID, time.Since(started).Round(time.Millisecond), err == nil)
	return err
}

func (w *ingestionWorker) runDenseStage(ctx context.Context, documentID string) error {
	s := w.service
	var doc indexDocument
	var storageKey, stage string
	var generation int64
	err := s.db.QueryRowContext(ctx, `UPDATE atlas_documents d SET status='processing',failure_reason=NULL,updated_at=now() FROM atlas_collections b WHERE d.id=$1::uuid AND b.id=d.collection_id AND d.status IN ('queued','processing') RETURNING b.namespace_id::text,b.id::text,d.filename,d.storage_key,b.index_version,d.index_stage,d.index_generation`, documentID).Scan(&doc.NamespaceID, &doc.CollectionID, &doc.Filename, &storageKey, &doc.IndexVersion, &stage, &generation)
	if errors.Is(err, sql.ErrNoRows) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("claim dense stage for document %s: %w", documentID, err)
	}
	if stage != "pending" {
		return nil
	}
	if doc.IndexVersion != s.indexVersion {
		return fmt.Errorf("document %s requires index version %s, but this service owns %s", documentID, doc.IndexVersion, s.indexVersion)
	}
	original, err := s.store.Open(ctx, storageKey)
	if err != nil {
		return fmt.Errorf("open original for document %s: %w", documentID, err)
	}
	processed, processErr := s.processor.RouteDocument(ctx, doc.Filename, original)
	closeErr := original.Close()
	if processErr != nil {
		return fmt.Errorf("route document %s: %w", documentID, processErr)
	}
	if closeErr != nil {
		return fmt.Errorf("close original for document %s: %w", documentID, closeErr)
	}
	if len(processed.Routes) == 0 || len(processed.Chunks) == 0 {
		return fmt.Errorf("processor returned no routes or chunks for document %s", documentID)
	}
	routeTexts := make([]string, len(processed.Routes))
	for index := range processed.Routes {
		routeTexts[index] = processed.Routes[index].Text
	}
	dense, err := s.embedder.Embed(ctx, routeTexts)
	if err != nil {
		return fmt.Errorf("embed routes for document %s: %w", documentID, err)
	}
	if len(dense) != len(processed.Routes) {
		return fmt.Errorf("TEI returned %d routes for document %s, expected %d", len(dense), documentID, len(processed.Routes))
	}
	for _, vector := range dense {
		if len(vector) != s.embeddingDimension {
			return fmt.Errorf("route embedding for document %s has %d dimensions, expected %d", documentID, len(vector), s.embeddingDimension)
		}
	}
	doc.DocumentID, doc.Routes, doc.Dense, doc.Chunks = documentID, processed.Routes, dense, processed.Chunks
	manifest, err := json.Marshal(processed.Chunks)
	if err != nil {
		return fmt.Errorf("encode chunk manifest for document %s: %w", documentID, err)
	}
	manifestKey := chunkManifestKey(storageKey)
	if err := s.store.Delete(ctx, manifestKey); err != nil {
		return fmt.Errorf("replace chunk manifest for document %s: %w", documentID, err)
	}
	if _, err := s.store.Put(ctx, manifestKey, bytes.NewReader(manifest), s.maxUploadBytes); err != nil {
		return fmt.Errorf("store chunk manifest for document %s: %w", documentID, err)
	}
	if err := s.index.ReplaceDenseRoutes(ctx, doc); err != nil {
		return fmt.Errorf("replace dense routes for document %s: %w", documentID, err)
	}
	return w.completeDenseAndQueueSparse(ctx, doc, generation)
}

func (w *ingestionWorker) completeDenseAndQueueSparse(ctx context.Context, doc indexDocument, generation int64) error {
	s := w.service
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("begin dense completion for document %s: %w", doc.DocumentID, err)
	}
	defer tx.Rollback()

	var openGeneration int64
	err = tx.QueryRowContext(ctx, `SELECT index_generation FROM atlas_collections WHERE id=$1::uuid FOR UPDATE`, doc.CollectionID).Scan(&openGeneration)
	if err == nil {
		_, err = tx.ExecContext(ctx, `UPDATE atlas_documents SET index_stage='dense',route_count=$2,chunk_count=$3,updated_at=now() WHERE id=$1::uuid AND index_stage='pending'`, doc.DocumentID, len(doc.Routes), len(doc.Chunks))
	}
	var pending int
	if err == nil {
		err = tx.QueryRowContext(ctx, `SELECT count(*) FROM atlas_documents WHERE collection_id=$1::uuid AND index_generation=$2 AND index_stage='pending'`, doc.CollectionID, generation).Scan(&pending)
	}
	if err == nil && pending == 0 && openGeneration == generation {
		_, err = tx.ExecContext(ctx, `UPDATE atlas_collections SET index_generation=index_generation+1,updated_at=now() WHERE id=$1::uuid`, doc.CollectionID)
		var rows *sql.Rows
		if err == nil {
			rows, err = tx.QueryContext(ctx, `SELECT id::text FROM atlas_documents WHERE collection_id=$1::uuid AND index_generation=$2 AND index_stage='dense' FOR UPDATE`, doc.CollectionID, generation)
		}
		var ids []string
		if err == nil {
			for rows.Next() {
				var id string
				if err = rows.Scan(&id); err != nil {
					break
				}
				ids = append(ids, id)
			}
			err = errors.Join(err, rows.Err(), rows.Close())
		}
		for _, id := range ids {
			if err != nil {
				break
			}
			_, err = s.jobs.InsertTx(ctx, tx, ingestionArgs{DocumentID: id, Stage: "sparse"}, &river.InsertOpts{MaxAttempts: s.maxAttempts, UniqueOpts: river.UniqueOpts{ByArgs: true}})
		}
	}
	if err == nil {
		err = tx.Commit()
	}
	if err != nil {
		return fmt.Errorf("complete dense stage for document %s: %w", doc.DocumentID, err)
	}
	return nil
}

func (w *ingestionWorker) runSparseStage(ctx context.Context, documentID string) error {
	s := w.service
	var doc indexDocument
	var storageKey, stage string
	var generation int64
	err := s.db.QueryRowContext(ctx, `UPDATE atlas_documents d SET status='processing',failure_reason=NULL,updated_at=now() FROM atlas_collections b WHERE d.id=$1::uuid AND b.id=d.collection_id AND d.status='processing' RETURNING b.namespace_id::text,b.id::text,d.filename,d.storage_key,b.index_version,d.index_stage,d.index_generation`, documentID).Scan(&doc.NamespaceID, &doc.CollectionID, &doc.Filename, &storageKey, &doc.IndexVersion, &stage, &generation)
	if errors.Is(err, sql.ErrNoRows) || err == nil && stage != "dense" {
		return nil
	}
	if err != nil {
		return fmt.Errorf("claim sparse stage for document %s: %w", documentID, err)
	}
	manifest, err := s.store.Open(ctx, chunkManifestKey(storageKey))
	if err != nil {
		return fmt.Errorf("open chunk manifest for document %s: %w", documentID, err)
	}
	decodeErr := json.NewDecoder(manifest).Decode(&doc.Chunks)
	closeErr := manifest.Close()
	if err = errors.Join(decodeErr, closeErr); err != nil {
		return fmt.Errorf("read chunk manifest for document %s: %w", documentID, err)
	}
	doc.Chunks, err = s.processor.SparseDocument(ctx, doc.Chunks)
	if err != nil {
		return fmt.Errorf("sparsify document %s: %w", documentID, err)
	}
	doc.DocumentID = documentID
	if err = s.index.ReplaceSparseChunks(ctx, doc); err != nil {
		return fmt.Errorf("replace sparse chunks for document %s: %w", documentID, err)
	}
	return w.completeSparseAndActivateGeneration(ctx, doc, generation)
}

func (w *ingestionWorker) completeSparseAndActivateGeneration(ctx context.Context, doc indexDocument, generation int64) error {
	s := w.service
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("begin sparse completion for document %s: %w", doc.DocumentID, err)
	}
	defer tx.Rollback()
	var locked int
	if err = tx.QueryRowContext(ctx, `SELECT 1 FROM atlas_collections WHERE id=$1::uuid FOR UPDATE`, doc.CollectionID).Scan(&locked); err == nil {
		_, err = tx.ExecContext(ctx, `UPDATE atlas_documents SET index_stage='sparse',updated_at=now() WHERE id=$1::uuid AND index_stage='dense'`, doc.DocumentID)
	}
	var pending int
	if err == nil {
		err = tx.QueryRowContext(ctx, `SELECT count(*) FROM atlas_documents WHERE collection_id=$1::uuid AND index_generation=$2 AND index_stage NOT IN ('sparse','ready')`, doc.CollectionID, generation).Scan(&pending)
	}
	var ids []string
	if err == nil && pending == 0 {
		rows, queryErr := tx.QueryContext(ctx, `SELECT id::text FROM atlas_documents WHERE collection_id=$1::uuid AND index_generation=$2 AND index_stage='sparse' FOR UPDATE`, doc.CollectionID, generation)
		if queryErr != nil {
			err = queryErr
		} else {
			for rows.Next() {
				var id string
				if err = rows.Scan(&id); err != nil {
					break
				}
				ids = append(ids, id)
			}
			err = errors.Join(err, rows.Err(), rows.Close())
		}
		if err == nil {
			err = s.index.ActivateDocuments(ctx, doc, ids)
		}
		if err == nil {
			_, err = tx.ExecContext(ctx, `UPDATE atlas_documents SET status='ready',index_stage='ready',failure_reason=NULL,updated_at=now() WHERE id=ANY($1::uuid[])`, pq.Array(ids))
		}
	}
	if err == nil {
		err = tx.Commit()
	}
	if err != nil {
		return fmt.Errorf("complete sparse stage for document %s: %w", doc.DocumentID, err)
	}
	return nil
}

func chunkManifestKey(storageKey string) string { return storageKey + ".chunks.json" }

func (h *ingestionErrorHandler) HandleError(ctx context.Context, job *rivertype.JobRow, workErr error) *river.ErrorHandlerResult {
	if job.Kind == (ingestionArgs{}).Kind() && job.Attempt >= job.MaxAttempts {
		h.markFailed(ctx, job, workErr.Error())
	}
	return nil
}

func (h *ingestionErrorHandler) HandlePanic(ctx context.Context, job *rivertype.JobRow, value any, _ string) *river.ErrorHandlerResult {
	if job.Kind == (ingestionArgs{}).Kind() && job.Attempt >= job.MaxAttempts {
		h.markFailed(ctx, job, fmt.Sprintf("worker panic: %v", value))
	}
	return nil
}

func (h *ingestionErrorHandler) markFailed(ctx context.Context, job *rivertype.JobRow, reason string) {
	var args ingestionArgs
	if err := json.Unmarshal(job.EncodedArgs, &args); err != nil || args.DocumentID == "" {
		log.Printf("cannot record exhausted knowledge job %d: invalid args: %v", job.ID, err)
		return
	}
	if _, err := h.db.ExecContext(ctx, `UPDATE atlas_documents SET status='failed',failure_reason=$2,updated_at=now() WHERE id=$1::uuid AND status='processing'`, args.DocumentID, reason); err != nil {
		log.Printf("cannot mark exhausted knowledge document %s failed: %v", args.DocumentID, err)
	}
}

func (p *processorClient) RouteDocument(ctx context.Context, filename string, src io.Reader) (processedDocument, error) {
	reader, writer := io.Pipe()
	multipartWriter := multipart.NewWriter(writer)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(p.url, "/")+"/route", reader)
	if err != nil {
		_ = reader.Close()
		_ = writer.Close()
		return processedDocument{}, err
	}
	req.Header.Set("Content-Type", multipartWriter.FormDataContentType())
	go func() {
		part, err := multipartWriter.CreateFormFile("file", filename)
		if err == nil {
			_, err = io.Copy(part, src)
		}
		if err == nil {
			err = multipartWriter.Close()
		}
		_ = writer.CloseWithError(err)
	}()
	var response processedDocument
	return response, p.sendAndDecode(req, &response)
}

func (p *processorClient) SparseDocument(ctx context.Context, chunks []processedChunk) ([]processedChunk, error) {
	var response struct {
		Chunks []processedChunk `json:"chunks"`
	}
	err := p.postJSON(ctx, "/sparse-document", map[string]any{"chunks": chunks}, &response)
	return response.Chunks, err
}

func (c modelClient) postJSON(ctx context.Context, path string, payload, target any) error {
	body, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(c.url, "/")+path, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	return c.sendAndDecode(req, target)
}

func (c modelClient) sendAndDecode(req *http.Request, target any) error {
	response, err := c.http.Do(req)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	if response.StatusCode/100 != 2 {
		message, _ := io.ReadAll(io.LimitReader(response.Body, 4096))
		return fmt.Errorf("%s returned %d: %s", c.name, response.StatusCode, strings.TrimSpace(string(message)))
	}
	return json.NewDecoder(response.Body).Decode(target)
}

func (t *teiClient) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	var vectors [][]float32
	if err := t.postJSON(ctx, "/embed", map[string]any{"inputs": texts, "truncate": false}, &vectors); err != nil {
		return nil, err
	}
	if len(vectors) != len(texts) {
		return nil, fmt.Errorf("TEI returned %d embeddings, expected %d", len(vectors), len(texts))
	}
	return vectors, nil
}

func (q *qdrantIndex) PrepareCollections(ctx context.Context) error {
	if err := q.ensureCollection(ctx, q.denseCollection, false); err != nil {
		return err
	}
	return q.ensureCollection(ctx, q.sparseCollection, true)
}

func (q *qdrantIndex) ensureCollection(ctx context.Context, name string, sparse bool) error {
	exists, err := q.client.CollectionExists(ctx, name)
	if err != nil {
		return fmt.Errorf("check Qdrant collection %s: %w", name, err)
	}
	if !exists {
		request := &qdrant.CreateCollection{CollectionName: name}
		if sparse {
			request.SparseVectorsConfig = qdrant.NewSparseVectorsConfig(map[string]*qdrant.SparseVectorParams{"bm25": {Modifier: qdrant.Modifier_Idf.Enum()}})
		} else {
			request.VectorsConfig = qdrant.NewVectorsConfig(&qdrant.VectorParams{Size: uint64(q.embeddingDimension), Distance: qdrant.Distance_Cosine})
		}
		if err := q.client.CreateCollection(ctx, request); err != nil {
			return fmt.Errorf("create Qdrant collection %s: %w", name, err)
		}
	}
	wait := true
	for _, field := range []struct {
		name string
		kind qdrant.FieldType
	}{
		{"namespace_id", qdrant.FieldType_FieldTypeKeyword}, {"collection_id", qdrant.FieldType_FieldTypeKeyword},
		{"document_id", qdrant.FieldType_FieldTypeKeyword}, {"index_version", qdrant.FieldType_FieldTypeKeyword}, {"active", qdrant.FieldType_FieldTypeBool},
	} {
		_, err := q.client.CreateFieldIndex(ctx, &qdrant.CreateFieldIndexCollection{CollectionName: name, FieldName: field.name, FieldType: field.kind.Enum(), Wait: &wait})
		if err != nil && status.Code(err) != codes.AlreadyExists {
			return fmt.Errorf("index Qdrant payload %s.%s: %w", name, field.name, err)
		}
	}
	return nil
}

func (q *qdrantIndex) ReplaceDenseRoutes(ctx context.Context, doc indexDocument) error {
	if len(doc.Routes) != len(doc.Dense) {
		return fmt.Errorf("dense routes and vectors differ")
	}
	basePayload := documentPayload(doc)
	densePoints := make([]*qdrant.PointStruct, len(doc.Routes))
	for index, route := range doc.Routes {
		payload := copyPayload(basePayload)
		payload["filename"], payload["facet_type"], payload["facet_text"] = doc.Filename, route.Type, route.Text
		values, err := qdrant.TryValueMap(payload)
		if err != nil {
			return fmt.Errorf("encode dense route %d: %w", index, err)
		}
		densePoints[index] = &qdrant.PointStruct{Id: deterministicPointID(doc.IndexVersion, doc.DocumentID, -index-1), Vectors: qdrant.NewVectorsDense(doc.Dense[index]), Payload: values}
	}
	// A dense retry starts the document index again, so stale chunks from an
	// earlier attempt cannot survive a shorter Docling result.
	if err := q.DeleteDocument(ctx, doc.NamespaceID, doc.CollectionID, doc.DocumentID); err != nil {
		return err
	}
	wait := true
	_, err := q.client.Upsert(ctx, &qdrant.UpsertPoints{CollectionName: q.denseCollection, Wait: &wait, Points: densePoints})
	if err != nil {
		return fmt.Errorf("write inactive dense routes: %w", err)
	}
	return nil
}

func (q *qdrantIndex) ReplaceSparseChunks(ctx context.Context, doc indexDocument) error {
	basePayload := documentPayload(doc)
	points := make([]*qdrant.PointStruct, 0, len(doc.Chunks))
	for _, chunk := range doc.Chunks {
		if len(chunk.Sparse.Indices) == 0 || len(chunk.Sparse.Indices) != len(chunk.Sparse.Values) {
			return fmt.Errorf("chunk %d has an invalid sparse vector", chunk.ChunkIndex)
		}
		payload := copyPayload(basePayload)
		payload["filename"], payload["text"], payload["chunk_index"] = doc.Filename, chunk.Text, chunk.ChunkIndex
		if chunk.Page != nil {
			payload["page"] = *chunk.Page
		}
		values, err := qdrant.TryValueMap(payload)
		if err != nil {
			return fmt.Errorf("encode sparse payload for chunk %d: %w", chunk.ChunkIndex, err)
		}
		points = append(points, &qdrant.PointStruct{Id: deterministicPointID(doc.IndexVersion, doc.DocumentID, chunk.ChunkIndex), Vectors: qdrant.NewVectorsMap(map[string]*qdrant.Vector{"bm25": qdrant.NewVectorSparse(chunk.Sparse.Indices, chunk.Sparse.Values)}), Payload: values})
	}
	if err := q.deleteFromCollection(ctx, q.sparseCollection, documentFilter(doc.NamespaceID, doc.CollectionID, doc.DocumentID)); err != nil {
		return err
	}
	wait := true
	for start := 0; start < len(points); start += qdrantBatchSize {
		end := min(start+qdrantBatchSize, len(points))
		if _, err := q.client.Upsert(ctx, &qdrant.UpsertPoints{CollectionName: q.sparseCollection, Wait: &wait, Points: points[start:end]}); err != nil {
			return fmt.Errorf("write inactive sparse point batch %d-%d: %w", start, end, err)
		}
	}
	return nil
}

func (q *qdrantIndex) ActivateDocuments(ctx context.Context, doc indexDocument, documentIDs []string) error {
	if len(documentIDs) == 0 {
		return errors.New("cannot activate an empty indexing generation")
	}
	filter := &qdrant.Filter{Must: []*qdrant.Condition{
		qdrant.NewMatchKeyword("namespace_id", doc.NamespaceID),
		qdrant.NewMatchKeyword("collection_id", doc.CollectionID),
		qdrant.NewMatchKeyword("index_version", doc.IndexVersion),
		qdrant.NewMatchKeywords("document_id", documentIDs...),
	}}
	selector := qdrant.NewPointsSelectorFilter(filter)
	wait := true
	for _, collection := range []string{q.denseCollection, q.sparseCollection} {
		if _, err := q.client.SetPayload(ctx, &qdrant.SetPayloadPoints{CollectionName: collection, Wait: &wait, Payload: qdrant.NewValueMap(map[string]any{"active": true}), PointsSelector: selector}); err != nil {
			return fmt.Errorf("activate document points in %s: %w", collection, err)
		}
	}
	return nil
}

func (q *qdrantIndex) DeleteDocument(ctx context.Context, namespaceID, collectionID, documentID string) error {
	for _, collection := range []string{q.denseCollection, q.sparseCollection} {
		if err := q.deleteFromCollection(ctx, collection, documentFilter(namespaceID, collectionID, documentID)); err != nil {
			return err
		}
	}
	return nil
}

func (q *qdrantIndex) deleteFromCollection(ctx context.Context, collection string, filter *qdrant.Filter) error {
	wait := true
	selector := qdrant.NewPointsSelectorFilter(filter)
	if _, err := q.client.Delete(ctx, &qdrant.DeletePoints{CollectionName: collection, Wait: &wait, Points: selector}); err != nil {
		return fmt.Errorf("delete document points from %s: %w", collection, err)
	}
	return nil
}

func documentPayload(doc indexDocument) map[string]any {
	return map[string]any{"namespace_id": doc.NamespaceID, "collection_id": doc.CollectionID, "document_id": doc.DocumentID, "index_version": doc.IndexVersion, "active": false}
}

func documentFilter(namespaceID, collectionID, documentID string) *qdrant.Filter {
	return &qdrant.Filter{Must: []*qdrant.Condition{qdrant.NewMatchKeyword("namespace_id", namespaceID), qdrant.NewMatchKeyword("collection_id", collectionID), qdrant.NewMatchKeyword("document_id", documentID)}}
}

func deterministicPointID(indexVersion, documentID string, chunk int) *qdrant.PointId {
	return qdrant.NewID(uuid.NewSHA1(uuid.NameSpaceOID, []byte(fmt.Sprintf("%s:%s:%d", indexVersion, documentID, chunk))).String())
}

func copyPayload(source map[string]any) map[string]any {
	copy := make(map[string]any, len(source)+4)
	for key, value := range source {
		copy[key] = value
	}
	return copy
}
