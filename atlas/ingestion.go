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

const (
	embeddingBatchSize = 8
	qdrantBatchSize    = 128
)

func (ingestionArgs) Kind() string { return "atlas_document_ingestion" }

func (w *ingestionWorker) Work(ctx context.Context, job *river.Job[ingestionArgs]) error {
	if job.Args.IndexVersion != "" && job.Args.IndexVersion != w.service.indexVersion {
		return nil
	}
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
	regions, processErr := s.processor.ProcessDocument(ctx, doc.Filename, original)
	closeErr := original.Close()
	if processErr != nil {
		return fmt.Errorf("process evidence regions for document %s: %w", documentID, processErr)
	}
	if closeErr != nil {
		return fmt.Errorf("close original for document %s: %w", documentID, closeErr)
	}
	if len(regions) == 0 {
		return fmt.Errorf("processor returned no evidence regions for document %s", documentID)
	}
	doc.DocumentID, doc.Regions = documentID, regions
	manifest, err := json.Marshal(regions)
	if err != nil {
		return fmt.Errorf("encode region manifest for document %s: %w", documentID, err)
	}
	manifestKey := regionManifestKey(storageKey)
	if err := s.store.Delete(ctx, manifestKey); err != nil {
		return fmt.Errorf("replace region manifest for document %s: %w", documentID, err)
	}
	if _, err := s.store.Put(ctx, manifestKey, bytes.NewReader(manifest), s.maxUploadBytes); err != nil {
		return fmt.Errorf("store region manifest for document %s: %w", documentID, err)
	}

	regionTexts := make([]string, len(regions))
	for index := range regions {
		regionTexts[index] = regions[index].Text
	}
	dense, err := s.embedder.Embed(ctx, regionTexts)
	if err != nil {
		return fmt.Errorf("embed evidence regions for document %s: %w", documentID, err)
	}
	if len(dense) != len(regions) {
		return fmt.Errorf("TEI returned %d regions for document %s, expected %d", len(dense), documentID, len(regions))
	}
	for _, vector := range dense {
		if len(vector) != s.embeddingDimension {
			return fmt.Errorf("region embedding for document %s has %d dimensions, expected %d", documentID, len(vector), s.embeddingDimension)
		}
	}
	doc.Dense = dense
	if err := s.index.ReplaceDenseRegions(ctx, doc); err != nil {
		return fmt.Errorf("replace dense regions for document %s: %w", documentID, err)
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
		_, err = tx.ExecContext(ctx, `UPDATE atlas_documents SET index_stage='dense',chunk_count=$2,updated_at=now() WHERE id=$1::uuid AND index_stage='pending'`, doc.DocumentID, len(doc.Regions))
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
			_, err = s.jobs.InsertTx(ctx, tx, ingestionArgs{DocumentID: id, Stage: "sparse", IndexVersion: doc.IndexVersion}, &river.InsertOpts{MaxAttempts: s.maxAttempts, UniqueOpts: river.UniqueOpts{ByArgs: true}})
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
	manifest, err := s.store.Open(ctx, regionManifestKey(storageKey))
	if err != nil {
		return fmt.Errorf("open region manifest for document %s: %w", documentID, err)
	}
	decodeErr := json.NewDecoder(manifest).Decode(&doc.Regions)
	closeErr := manifest.Close()
	if err = errors.Join(decodeErr, closeErr); err != nil {
		return fmt.Errorf("read region manifest for document %s: %w", documentID, err)
	}
	doc.Regions, err = s.processor.SparseDocument(ctx, doc.Regions)
	if err != nil {
		return fmt.Errorf("sparsify document %s: %w", documentID, err)
	}
	doc.DocumentID = documentID
	if err = s.index.ReplaceSparseRegions(ctx, doc); err != nil {
		return fmt.Errorf("replace sparse regions for document %s: %w", documentID, err)
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

func regionManifestKey(storageKey string) string { return storageKey + ".regions.json" }

func legacyChunkManifestKey(storageKey string) string { return storageKey + ".chunks.json" }

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

func (p *processorClient) ProcessDocument(ctx context.Context, filename string, src io.Reader) ([]processedRegion, error) {
	reader, writer := io.Pipe()
	multipartWriter := multipart.NewWriter(writer)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(p.url, "/")+"/process", reader)
	if err != nil {
		_ = reader.Close()
		_ = writer.Close()
		return nil, err
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
	var response struct {
		Regions []processedRegion `json:"regions"`
	}
	err = p.sendAndDecode(req, &response)
	return response.Regions, err
}

func (p *processorClient) SparseDocument(ctx context.Context, regions []processedRegion) ([]processedRegion, error) {
	var response struct {
		Regions []processedRegion `json:"regions"`
	}
	err := p.postJSON(ctx, "/sparse-document", map[string]any{"regions": regions}, &response)
	return response.Regions, err
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
	vectors := make([][]float32, 0, len(texts))
	for start := 0; start < len(texts); start += embeddingBatchSize {
		end := min(start+embeddingBatchSize, len(texts))
		var batch [][]float32
		if err := t.postJSON(ctx, "/embed", map[string]any{"inputs": texts[start:end], "truncate": false}, &batch); err != nil {
			return nil, err
		}
		if len(batch) != end-start {
			return nil, fmt.Errorf("TEI returned %d embeddings for batch %d-%d, expected %d", len(batch), start, end, end-start)
		}
		vectors = append(vectors, batch...)
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
		{"document_id", qdrant.FieldType_FieldTypeKeyword}, {"region_id", qdrant.FieldType_FieldTypeKeyword}, {"index_version", qdrant.FieldType_FieldTypeKeyword}, {"active", qdrant.FieldType_FieldTypeBool},
	} {
		_, err := q.client.CreateFieldIndex(ctx, &qdrant.CreateFieldIndexCollection{CollectionName: name, FieldName: field.name, FieldType: field.kind.Enum(), Wait: &wait})
		if err != nil && status.Code(err) != codes.AlreadyExists {
			return fmt.Errorf("index Qdrant payload %s.%s: %w", name, field.name, err)
		}
	}
	if sparse {
		lowercase, phrases := true, true
		_, err := q.client.CreateFieldIndex(ctx, &qdrant.CreateFieldIndexCollection{
			CollectionName: name, FieldName: "text", FieldType: qdrant.FieldType_FieldTypeText.Enum(), Wait: &wait,
			FieldIndexParams: &qdrant.PayloadIndexParams{IndexParams: &qdrant.PayloadIndexParams_TextIndexParams{TextIndexParams: &qdrant.TextIndexParams{Tokenizer: qdrant.TokenizerType_Word, Lowercase: &lowercase, PhraseMatching: &phrases}}},
		})
		if err != nil && status.Code(err) != codes.AlreadyExists {
			return fmt.Errorf("index Qdrant evidence text: %w", err)
		}
	}
	return nil
}

func (q *qdrantIndex) ReplaceDenseRegions(ctx context.Context, doc indexDocument) error {
	if len(doc.Regions) != len(doc.Dense) {
		return fmt.Errorf("dense regions and vectors differ")
	}
	basePayload := documentPayload(doc)
	densePoints := make([]*qdrant.PointStruct, len(doc.Regions))
	for index, region := range doc.Regions {
		payload := copyPayload(basePayload)
		pointID := deterministicPointID(doc.IndexVersion, doc.DocumentID, region.RegionIndex)
		payload["region_id"], payload["region_index"] = pointID.GetUuid(), region.RegionIndex
		if region.Page != nil {
			payload["page"] = *region.Page
		}
		values, err := qdrant.TryValueMap(payload)
		if err != nil {
			return fmt.Errorf("encode dense region %d: %w", index, err)
		}
		densePoints[index] = &qdrant.PointStruct{Id: pointID, Vectors: qdrant.NewVectorsDense(doc.Dense[index]), Payload: values}
	}
	// A dense retry starts the document index again, so stale regions from an
	// earlier attempt cannot survive a shorter Docling result.
	if err := q.DeleteDocument(ctx, doc.NamespaceID, doc.CollectionID, doc.DocumentID); err != nil {
		return err
	}
	wait := true
	for start := 0; start < len(densePoints); start += qdrantBatchSize {
		end := min(start+qdrantBatchSize, len(densePoints))
		if _, err := q.client.Upsert(ctx, &qdrant.UpsertPoints{CollectionName: q.denseCollection, Wait: &wait, Points: densePoints[start:end]}); err != nil {
			return fmt.Errorf("write inactive dense region batch %d-%d: %w", start, end, err)
		}
	}
	return nil
}

func (q *qdrantIndex) ReplaceSparseRegions(ctx context.Context, doc indexDocument) error {
	basePayload := documentPayload(doc)
	points := make([]*qdrant.PointStruct, 0, len(doc.Regions))
	for _, region := range doc.Regions {
		if len(region.Sparse.Indices) == 0 || len(region.Sparse.Indices) != len(region.Sparse.Values) {
			return fmt.Errorf("region %d has an invalid sparse vector", region.RegionIndex)
		}
		payload := copyPayload(basePayload)
		pointID := deterministicPointID(doc.IndexVersion, doc.DocumentID, region.RegionIndex)
		payload["filename"], payload["text"], payload["region_id"], payload["chunk_index"] = doc.Filename, region.Text, pointID.GetUuid(), region.RegionIndex
		if region.Page != nil {
			payload["page"] = *region.Page
		}
		values, err := qdrant.TryValueMap(payload)
		if err != nil {
			return fmt.Errorf("encode sparse payload for region %d: %w", region.RegionIndex, err)
		}
		points = append(points, &qdrant.PointStruct{Id: pointID, Vectors: qdrant.NewVectorsMap(map[string]*qdrant.Vector{"bm25": qdrant.NewVectorSparse(region.Sparse.Indices, region.Sparse.Values)}), Payload: values})
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

func deterministicPointID(indexVersion, documentID string, region int) *qdrant.PointId {
	return qdrant.NewID(uuid.NewSHA1(uuid.NameSpaceOID, []byte(fmt.Sprintf("%s:%s:%d", indexVersion, documentID, region))).String())
}

func copyPayload(source map[string]any) map[string]any {
	copy := make(map[string]any, len(source)+4)
	for key, value := range source {
		copy[key] = value
	}
	return copy
}
