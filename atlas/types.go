package atlas

import (
	"context"
	"database/sql"
	"io"
	"net/http"
	"time"

	"github.com/minio/minio-go/v7"
	"github.com/qdrant/go-client/qdrant"
	"github.com/riverqueue/river"
)

type Config struct {
	Addr, DatabaseURL, ServiceToken string
}

type Collection struct {
	ID           string    `json:"id"`
	Name         string    `json:"name"`
	Description  string    `json:"description"`
	IndexVersion string    `json:"index_version"`
	CreatedAt    time.Time `json:"created_at"`
}

type Document struct {
	ID            string    `json:"id"`
	CollectionID  string    `json:"collection_id"`
	Filename      string    `json:"filename"`
	MimeType      string    `json:"mime_type"`
	SHA256        string    `json:"sha256"`
	Status        string    `json:"status"`
	IndexStage    string    `json:"index_stage"`
	SizeBytes     int64     `json:"size_bytes"`
	FailureReason *string   `json:"failure_reason"`
	ChunkCount    int       `json:"chunk_count"`
	CreatedAt     time.Time `json:"created_at"`
}

type Citation struct {
	CitationID string `json:"citation_id"`
	DocumentID string `json:"document_id"`
	Filename   string `json:"filename"`
	Page       *int   `json:"page"`
	ChunkIndex int    `json:"chunk_index"`
}

type Candidate struct {
	Citation
	Text  string  `json:"text"`
	Score float32 `json:"score"`
}

type objectStore interface {
	Put(context.Context, string, io.Reader, int64) (storedObject, error)
	Open(context.Context, string) (io.ReadCloser, error)
	Delete(context.Context, string) error
}

type documentProcessor interface {
	ProcessDocument(context.Context, string, io.Reader) ([]processedRegion, error)
	SparseDocument(context.Context, []processedRegion) ([]processedRegion, error)
	SparseQuery(context.Context, string) (sparseVector, error)
}

type denseEmbedder interface {
	Embed(context.Context, []string) ([][]float32, error)
}

type vectorIndex interface {
	PrepareCollections(context.Context) error
	ReplaceDenseRegions(context.Context, indexDocument) error
	ReplaceSparseRegions(context.Context, indexDocument) error
	ActivateDocuments(context.Context, indexDocument, []string) error
	DeleteDocument(context.Context, string, string, string) error
	SearchDense(context.Context, retrievalScope, []float32, uint64) ([]string, error)
	SearchSparse(context.Context, retrievalScope, sparseVector, []string, uint64) ([]Candidate, error)
}

type Service struct {
	db                              *sql.DB
	store                           objectStore
	processor                       documentProcessor
	embedder                        denseEmbedder
	index                           vectorIndex
	jobs                            *river.Client[*sql.Tx]
	serviceToken, indexVersion      string
	queryPrefix                     string
	embeddingDimension, maxAttempts int
	maxUploadBytes                  int64
	denseThreshold, sparseThreshold float32
}

type clientError struct {
	status  int
	message string
}

type ingestionArgs struct {
	DocumentID   string `json:"document_id"`
	Stage        string `json:"stage"`
	IndexVersion string `json:"index_version"`
}

type sparseVector struct {
	Indices []uint32  `json:"indices"`
	Values  []float32 `json:"values"`
}

type processedRegion struct {
	Text        string       `json:"text"`
	Page        *int         `json:"page"`
	RegionIndex int          `json:"region_index"`
	Sparse      sparseVector `json:"sparse,omitempty"`
}

type indexDocument struct {
	NamespaceID, CollectionID, DocumentID, Filename, IndexVersion string
	Dense                                                         [][]float32
	Regions                                                       []processedRegion
}

type ingestionWorker struct {
	river.WorkerDefaults[ingestionArgs]
	service *Service
}

type ingestionErrorHandler struct{ db *sql.DB }

type modelClient struct {
	url, name string
	http      *http.Client
}

type processorClient struct{ modelClient }

type teiClient struct{ modelClient }

type qdrantIndex struct {
	client                            *qdrant.Client
	denseCollection, sparseCollection string
	embeddingDimension                int
}

type retrievalScope struct {
	NamespaceID, CollectionID, IndexVersion string
	DenseThreshold, SparseThreshold         float32
}

type storedObject struct {
	SHA256 string
	Size   int64
}

type localStore struct{ root string }

type s3Store struct {
	client *minio.Client
	bucket string
}
