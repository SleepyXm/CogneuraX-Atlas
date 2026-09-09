package atlas

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/DATA-DOG/go-sqlmock"
	"github.com/riverqueue/river"
	"github.com/riverqueue/river/riverdriver/riverdatabasesql"
)

type fakeIndex struct {
	dense, regions            []string
	anchors                   []string
	scoped                    []Candidate
	denseLimit, denseCalls    uint64
	denseWrites, sparseWrites int
	denseRegions, activations int
	denseError, sparseError   error
	sparseCalls               int
}

func (*fakeIndex) PrepareCollections(context.Context) error { return nil }
func (f *fakeIndex) ReplaceDenseRegions(_ context.Context, doc indexDocument) error {
	f.denseWrites++
	f.denseRegions = len(doc.Dense)
	return f.denseError
}
func (f *fakeIndex) ReplaceSparseRegions(context.Context, indexDocument) error {
	f.sparseWrites++
	return f.sparseError
}
func (f *fakeIndex) ActivateDocuments(_ context.Context, _ indexDocument, ids []string) error {
	f.activations += len(ids)
	return nil
}
func (*fakeIndex) DeleteDocument(context.Context, string, string, string) error { return nil }
func (f *fakeIndex) SearchDense(_ context.Context, _ retrievalScope, _ []float32, limit uint64) ([]string, error) {
	f.denseLimit = limit
	f.denseCalls++
	return f.regions, nil
}
func (f *fakeIndex) SearchSparse(_ context.Context, _ retrievalScope, _ sparseVector, regions, anchors []string, _ uint64) ([]Candidate, error) {
	f.sparseCalls++
	f.dense = append([]string(nil), regions...)
	f.anchors = append([]string(nil), anchors...)
	return f.scoped, nil
}

type fakeProcessor struct {
	regions    []processedRegion
	processErr error
	sparseErr  error
	queryCalls *int
	queryText  *string
}

func (f fakeProcessor) ProcessDocument(context.Context, string, io.Reader) ([]processedRegion, error) {
	return f.regions, f.processErr
}
func (f fakeProcessor) SparseDocument(_ context.Context, regions []processedRegion) ([]processedRegion, error) {
	for index := range regions {
		regions[index].Sparse = sparseVector{Indices: []uint32{1}, Values: []float32{1}}
	}
	return regions, f.sparseErr
}
func (f fakeProcessor) SparseQuery(_ context.Context, query string) (sparseVector, error) {
	if f.queryCalls != nil {
		*f.queryCalls++
	}
	if f.queryText != nil {
		*f.queryText = query
	}
	return sparseVector{Indices: []uint32{1}, Values: []float32{1}}, nil
}

func TestPromptPassSeparatesOriginalSpans(t *testing.T) {
	cases := []struct {
		question string
		dense    string
		sparse   string
		anchors  []string
	}{
		{"Who challenged plague theory?", "Who challenged plague theory?", "challenged plague theory", nil},
		{"According to the 2000 United States Census, how many people were living in Atlantic City?", "According to the 2000 United States Census, how many people were living in Atlantic City?", "2000 United States Census people living Atlantic City", []string{"2000", "United States Census", "Atlantic City"}},
		{"Who has no power to pass laws?", "Who has no power to pass laws?", "no power pass laws", nil},
		{"Who did John B Watson and David Graeber discover a fossil of?", "Who did John B Watson and David Graeber discover a fossil of?", "John B Watson David Graeber discover fossil", []string{"John B Watson", "David Graeber"}},
		{"Where was France's Huguenot population centered?", "Where was France's Huguenot population centered?", "France Huguenot population centered", nil},
		{"What happened in March of 1974?", "What happened in March of 1974?", "happened March 1974", []string{"1974"}},
		{"What makes up the European Union's legislature?", "What makes up the European Union's legislature?", "makes up European Union legislature", []string{"European Union"}},
	}
	for _, test := range cases {
		pass := partitionPrompt(test.question)
		if pass.Dense != test.dense || pass.Sparse != test.sparse || !slices.Equal(pass.Anchors, test.anchors) {
			t.Fatalf("unexpected prompt pass for %q: %+v", test.question, pass)
		}
	}
}

func TestRetrievalUsesSeparatedPromptPass(t *testing.T) {
	db, mock, _ := sqlmock.New()
	defer db.Close()
	namespaceID, collectionID, documentID := "00000000-0000-0000-0000-000000000001", "00000000-0000-0000-0000-000000000002", "00000000-0000-0000-0000-000000000003"
	mock.ExpectQuery("SELECT b.index_version").WillReturnRows(sqlmock.NewRows([]string{"version", "regions"}).AddRow("v2", 8))
	mock.ExpectQuery("SELECT d.id::text").WillReturnRows(sqlmock.NewRows([]string{"id", "filename"}).AddRow(documentID, "doc.md"))
	index := &fakeIndex{regions: []string{"c"}, scoped: []Candidate{{Citation: Citation{CitationID: "c", DocumentID: documentID}, Text: "answer"}}}
	sparseText, denseText := "", []string{}
	service := &Service{db: db, processor: fakeProcessor{queryText: &sparseText}, embedder: fakeEmbedder{inputs: &denseText}, index: index, indexVersion: "v2", queryPrefix: "query: ", denseThreshold: .5, sparseThreshold: 12.5}
	results, err := service.retrieveEvidenceRegions(context.Background(), namespaceID, collectionID, "According to the 2000 United States Census, how many people were living in Atlantic City?")
	if err != nil || len(results) != 1 || index.denseCalls != 1 || !slices.Equal(denseText, []string{"query: According to the 2000 United States Census, how many people were living in Atlantic City?"}) || sparseText != "2000 United States Census people living Atlantic City" || !slices.Equal(index.anchors, []string{"2000", "United States Census", "Atlantic City"}) {
		t.Fatalf("prompt was not separated: results=%v index=%+v dense=%v sparse=%q err=%v", results, index, denseText, sparseText, err)
	}
}

type fakeEmbedder struct {
	vector [][]float32
	err    error
	inputs *[]string
}

func (f fakeEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	if f.inputs != nil {
		*f.inputs = append([]string(nil), texts...)
	}
	if f.vector == nil {
		f.vector = make([][]float32, len(texts))
		for index := range texts {
			f.vector[index] = []float32{1}
		}
	}
	return f.vector, f.err
}

type fakeStore struct{ manifest string }

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) { return f(request) }

func (fakeStore) Put(context.Context, string, io.Reader, int64) (storedObject, error) {
	return storedObject{SHA256: strings.Repeat("a", 64), Size: 4}, nil
}
func (f fakeStore) Open(_ context.Context, key string) (io.ReadCloser, error) {
	if strings.HasSuffix(key, ".regions.json") {
		return io.NopCloser(strings.NewReader(f.manifest)), nil
	}
	return io.NopCloser(strings.NewReader("document")), nil
}
func (fakeStore) Delete(context.Context, string) error { return nil }

func TestProcessorClientReturnsDecodedRegions(t *testing.T) {
	httpClient := &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		if request.URL.Path != "/process" {
			t.Errorf("unexpected processor path %s", request.URL.Path)
		}
		_, _ = io.Copy(io.Discard, request.Body)
		return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader(`{"regions":[{"text":"evidence"}]}`)), Header: make(http.Header)}, nil
	})}
	client := &processorClient{modelClient{url: "http://processor", name: "processor", http: httpClient}}
	regions, err := client.ProcessDocument(context.Background(), "doc.md", strings.NewReader("source"))
	if err != nil || len(regions) != 1 || regions[0].Text != "evidence" {
		t.Fatalf("processor response was not returned: regions=%v err=%v", regions, err)
	}
}

func TestTEIClientBatchesRegionEmbeddings(t *testing.T) {
	calls := 0
	httpClient := &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		var body struct {
			Inputs []string `json:"inputs"`
		}
		_ = json.NewDecoder(request.Body).Decode(&body)
		calls++
		vectors := make([][]float32, len(body.Inputs))
		encoded, _ := json.Marshal(vectors)
		return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(bytes.NewReader(encoded)), Header: make(http.Header)}, nil
	})}
	client := &teiClient{modelClient{url: "http://tei", name: "TEI", http: httpClient}}
	vectors, err := client.Embed(context.Background(), make([]string, embeddingBatchSize+1))
	if err != nil || len(vectors) != embeddingBatchSize+1 || calls != 2 {
		t.Fatalf("TEI batching changed: calls=%d vectors=%d err=%v", calls, len(vectors), err)
	}
}

func TestDocumentUploadStoresMetadataAndRiverJobTogether(t *testing.T) {
	db, mock, _ := sqlmock.New()
	defer db.Close()
	now := time.Now()
	collectionID := "00000000-0000-0000-0000-000000000002"
	mock.ExpectBegin()
	mock.ExpectQuery("INSERT INTO atlas_documents").WillReturnRows(sqlmock.NewRows([]string{"id", "base", "filename", "mime", "size", "sha", "status", "stage", "failure", "chunks", "created"}).AddRow("00000000-0000-0000-0000-000000000003", collectionID, "doc.md", "text/markdown", 4, strings.Repeat("a", 64), "queued", "pending", nil, 0, now))
	mock.ExpectQuery("INSERT INTO .*river_job").WillReturnRows(sqlmock.NewRows([]string{"id", "args", "attempt", "attempted_at", "attempted_by", "created_at", "errors", "finalized_at", "kind", "max_attempts", "metadata", "priority", "queue", "state", "scheduled_at", "tags", "unique_key", "unique_states", "duplicate"}).AddRow(1, `{}`, 0, nil, "{}", now, "{}", nil, ingestionArgs{}.Kind(), 3, `{}`, 1, "default", "available", now, "{}", nil, nil, false))
	mock.ExpectExec("SELECT pg_notify").WillReturnResult(sqlmock.NewResult(0, 1))
	mock.ExpectCommit()
	jobs, _ := river.NewClient(riverdatabasesql.New(db), &river.Config{})
	service := &Service{db: db, store: fakeStore{}, jobs: jobs, serviceToken: "secret", maxAttempts: 3, maxUploadBytes: 100}

	var body bytes.Buffer
	form := multipart.NewWriter(&body)
	file, _ := form.CreateFormFile("file", "doc.md")
	_, _ = file.Write([]byte("data"))
	_ = form.Close()
	request := httptest.NewRequest(http.MethodPost, "/v1/collections/"+collectionID+"/documents", &body)
	request.Header.Set("Content-Type", form.FormDataContentType())
	request.Header.Set("Authorization", "Bearer secret")
	request.Header.Set("X-Atlas-Namespace-ID", "00000000-0000-0000-0000-000000000001")
	response := httptest.NewRecorder()
	service.Router().ServeHTTP(response, request)
	if response.Code != http.StatusAccepted {
		t.Fatalf("upload returned %d: %s", response.Code, response.Body.String())
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatal(err)
	}
}

func TestDocumentListReturnsOneBoundedPage(t *testing.T) {
	db, mock, _ := sqlmock.New()
	defer db.Close()
	namespaceID, collectionID := "00000000-0000-0000-0000-000000000001", "00000000-0000-0000-0000-000000000002"
	rows := sqlmock.NewRows([]string{"id", "base", "filename", "mime", "size", "sha", "status", "stage", "failure", "chunks", "created"})
	for index := 1; index <= 3; index++ {
		rows.AddRow(fmt.Sprintf("00000000-0000-0000-0000-%012d", index), collectionID, fmt.Sprintf("%d.md", index), "text/markdown", 4, strings.Repeat("a", 64), "ready", "ready", nil, 1, time.Now())
	}
	mock.ExpectQuery("LIMIT \\$3 OFFSET \\$4").WithArgs(namespaceID, collectionID, 3, 5).WillReturnRows(rows)
	service := &Service{db: db, serviceToken: "secret"}
	request := httptest.NewRequest(http.MethodGet, "/v1/collections/"+collectionID+"/documents?limit=2&offset=5", nil)
	request.Header.Set("Authorization", "Bearer secret")
	request.Header.Set("X-Atlas-Namespace-ID", namespaceID)
	response := httptest.NewRecorder()
	service.Router().ServeHTTP(response, request)
	var page struct {
		Data       []Document `json:"data"`
		NextOffset int        `json:"next_offset"`
	}
	if err := json.Unmarshal(response.Body.Bytes(), &page); err != nil || response.Code != http.StatusOK || len(page.Data) != 2 || page.NextOffset != 7 {
		t.Fatalf("unexpected document page: status=%d body=%s err=%v", response.Code, response.Body.String(), err)
	}
}

func TestRetrievalCasesStayInsideOneWorkflow(t *testing.T) {
	namespaceID, collectionID, documentID := "00000000-0000-0000-0000-000000000001", "00000000-0000-0000-0000-000000000002", "00000000-0000-0000-0000-000000000003"
	candidate := Candidate{Citation: Citation{CitationID: "chunk", DocumentID: documentID}, Text: "answer"}
	cases := []struct {
		name            string
		regions         []string
		scoped          []Candidate
		wantResults     int
		wantSparseCalls int
	}{
		{"dense and sparse match", []string{"region-a"}, []Candidate{candidate}, 1, 1},
		{"sparse miss stays empty", []string{"region-a"}, nil, 0, 1},
		{"dense miss stays empty", nil, nil, 0, 0},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			db, mock, _ := sqlmock.New()
			defer db.Close()
			mock.ExpectQuery("SELECT b.index_version").WillReturnRows(sqlmock.NewRows([]string{"version", "regions"}).AddRow("v2", 8))
			if test.wantResults > 0 {
				mock.ExpectQuery("SELECT d.id::text").WillReturnRows(sqlmock.NewRows([]string{"id", "filename"}).AddRow(documentID, "doc.md"))
			}
			index, queryCalls := &fakeIndex{regions: test.regions, scoped: test.scoped}, 0
			service := &Service{db: db, processor: fakeProcessor{queryCalls: &queryCalls}, embedder: fakeEmbedder{}, index: index, indexVersion: "v2", queryPrefix: "query: ", denseThreshold: .55, sparseThreshold: 5}
			results, err := service.retrieveEvidenceRegions(context.Background(), namespaceID, collectionID, "question")
			if err != nil || len(results) != test.wantResults || index.denseLimit != 8 || index.sparseCalls != test.wantSparseCalls || queryCalls != test.wantSparseCalls || test.wantSparseCalls > 0 && !slices.Equal(index.dense, test.regions) {
				t.Fatalf("retrieval flow changed: results=%v index=%+v err=%v", results, index, err)
			}
			if err := mock.ExpectationsWereMet(); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestIngestionPassesFailEarly(t *testing.T) {
	regions := []processedRegion{{Text: "first region"}, {Text: "second region", RegionIndex: 1}}
	cases := []struct {
		name                string
		processor           fakeProcessor
		embedder            fakeEmbedder
		indexError          error
		wantError           bool
		expectedDenseWrites int
	}{
		{"processor", fakeProcessor{regions: regions, processErr: errors.New("processor failed")}, fakeEmbedder{}, nil, true, 0},
		{"embedding", fakeProcessor{regions: regions}, fakeEmbedder{err: errors.New("embedding failed")}, nil, true, 0},
		{"dimension", fakeProcessor{regions: regions}, fakeEmbedder{vector: [][]float32{{1, 2}}}, nil, true, 0},
		{"index", fakeProcessor{regions: regions}, fakeEmbedder{}, errors.New("index failed"), true, 1},
		{"success", fakeProcessor{regions: regions}, fakeEmbedder{}, nil, false, 1},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			db, mock, _ := sqlmock.New()
			defer db.Close()
			mock.ExpectQuery("UPDATE atlas_documents").WillReturnRows(sqlmock.NewRows([]string{"user", "base", "filename", "key", "version", "stage", "generation"}).AddRow("user", "base", "doc.md", "key", "v1", "pending", 1))
			if !test.wantError {
				mock.ExpectBegin()
				mock.ExpectQuery("SELECT index_generation").WillReturnRows(sqlmock.NewRows([]string{"generation"}).AddRow(1))
				mock.ExpectExec("UPDATE atlas_documents SET index_stage='dense'").WillReturnResult(sqlmock.NewResult(0, 1))
				mock.ExpectQuery("SELECT count").WillReturnRows(sqlmock.NewRows([]string{"count"}).AddRow(1))
				mock.ExpectCommit()
			}
			index := &fakeIndex{denseError: test.indexError}
			service := &Service{db: db, store: fakeStore{}, processor: test.processor, embedder: test.embedder, index: index, indexVersion: "v1", embeddingDimension: 1}
			err := (&ingestionWorker{service: service}).Work(context.Background(), &river.Job[ingestionArgs]{Args: ingestionArgs{DocumentID: "doc"}})
			if (err != nil) != test.wantError || index.denseWrites != test.expectedDenseWrites || (!test.wantError && index.denseRegions != 2) {
				t.Fatalf("unexpected pass result: dense_writes=%d err=%v", index.denseWrites, err)
			}
			if err := mock.ExpectationsWereMet(); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestSparseStageActivatesOnlyCompleteGeneration(t *testing.T) {
	db, mock, _ := sqlmock.New()
	defer db.Close()
	documentID := "00000000-0000-0000-0000-000000000003"
	manifest, _ := json.Marshal([]processedRegion{{Text: "region", RegionIndex: 0}})
	mock.ExpectQuery("UPDATE atlas_documents").WillReturnRows(sqlmock.NewRows([]string{"user", "base", "filename", "key", "version", "stage", "generation"}).AddRow("user", "base", "doc.md", "key", "v1", "dense", 1))
	mock.ExpectBegin()
	mock.ExpectQuery("SELECT 1 FROM atlas_collections").WillReturnRows(sqlmock.NewRows([]string{"lock"}).AddRow(1))
	mock.ExpectExec("UPDATE atlas_documents SET index_stage='sparse'").WillReturnResult(sqlmock.NewResult(0, 1))
	mock.ExpectQuery("SELECT count").WillReturnRows(sqlmock.NewRows([]string{"count"}).AddRow(0))
	mock.ExpectQuery("SELECT id::text").WillReturnRows(sqlmock.NewRows([]string{"id"}).AddRow(documentID))
	mock.ExpectExec("UPDATE atlas_documents SET status='ready'").WillReturnResult(sqlmock.NewResult(0, 1))
	mock.ExpectCommit()
	index := &fakeIndex{}
	service := &Service{db: db, store: fakeStore{manifest: string(manifest)}, processor: fakeProcessor{}, index: index}
	err := (&ingestionWorker{service: service}).Work(context.Background(), &river.Job[ingestionArgs]{Args: ingestionArgs{DocumentID: documentID, Stage: "sparse"}})
	if err != nil || index.sparseWrites != 1 || index.activations != 1 {
		t.Fatalf("unexpected sparse stage: index=%+v err=%v", index, err)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatal(err)
	}
}
