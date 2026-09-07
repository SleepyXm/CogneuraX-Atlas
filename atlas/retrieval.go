package atlas

import (
	"context"
	"fmt"
	"strings"

	"github.com/lib/pq"
	"github.com/qdrant/go-client/qdrant"
)

func (p *processorClient) SparseQuery(ctx context.Context, query string) (sparseVector, error) {
	var vector sparseVector
	err := p.postJSON(ctx, "/sparse-query", map[string]string{"query": query}, &vector)
	return vector, err
}

func (s *Service) retrieveRelevantChunks(ctx context.Context, namespaceID, collectionID, question string) ([]Candidate, error) {
	// 1. Authorize through PostgreSQL and size searches from ready content.
	var indexVersion string
	var readyDocuments, readyChunks int
	err := s.db.QueryRowContext(ctx, `SELECT b.index_version,count(d.id),coalesce(sum(d.chunk_count),0) FROM atlas_collections b LEFT JOIN atlas_documents d ON d.collection_id=b.id AND d.status='ready' WHERE b.id=$1::uuid AND b.namespace_id=$2::uuid GROUP BY b.index_version`, collectionID, namespaceID).Scan(&indexVersion, &readyDocuments, &readyChunks)
	if err != nil {
		return nil, err
	}
	if indexVersion != s.indexVersion || readyDocuments == 0 || readyChunks == 0 {
		return []Candidate{}, nil
	}

	// 2. Encode the question once for each retrieval model.
	dense, err := s.embedder.Embed(ctx, []string{s.queryPrefix + question})
	if err != nil {
		return nil, fmt.Errorf("embed retrieval question: %w", err)
	}
	sparse, err := s.processor.SparseQuery(ctx, question)
	if err != nil {
		return nil, fmt.Errorf("encode sparse retrieval question: %w", err)
	}
	scope := retrievalScope{NamespaceID: namespaceID, CollectionID: collectionID, IndexVersion: indexVersion, DenseThreshold: s.denseThreshold, SparseThreshold: s.sparseThreshold}
	// 3. Dense profile similarity gates the eligible documents.
	documentIDs, err := s.index.SearchDense(ctx, scope, dense[0], uint64(readyDocuments*denseRoutesPerDocument))
	if err != nil {
		return nil, fmt.Errorf("dense document gate: %w", err)
	}

	// 4. Search detailed chunks inside the dense matches.
	var candidates []Candidate
	if len(documentIDs) > 0 {
		candidates, err = s.index.SearchSparse(ctx, scope, sparse, documentIDs, uint64(readyChunks))
		if err != nil {
			return nil, fmt.Errorf("sparse search inside dense matches: %w", err)
		}
	}
	// 5. Only an empty scoped result permits the whole-base sparse fallback.
	if len(candidates) == 0 {
		candidates, err = s.index.SearchSparse(ctx, scope, sparse, nil, uint64(readyChunks))
		if err != nil {
			return nil, fmt.Errorf("collection sparse fallback: %w", err)
		}
	}
	// 6. Reject stale Qdrant points while preserving sparse-score order.
	return s.validateReadyCandidates(ctx, namespaceID, collectionID, indexVersion, candidates)
}

func (s *Service) validateReadyCandidates(ctx context.Context, namespaceID, collectionID, indexVersion string, candidates []Candidate) ([]Candidate, error) {
	if len(candidates) == 0 {
		return []Candidate{}, nil
	}
	ids := make([]string, 0, len(candidates))
	for _, candidate := range candidates {
		ids = append(ids, candidate.DocumentID)
	}
	rows, err := s.db.QueryContext(ctx, `SELECT d.id::text,d.filename FROM atlas_documents d JOIN atlas_collections b ON b.id=d.collection_id WHERE b.namespace_id=$1::uuid AND b.id=$2::uuid AND b.index_version=$3 AND d.status='ready' AND d.id=ANY($4::uuid[])`, namespaceID, collectionID, indexVersion, pq.Array(ids))
	if err != nil {
		return nil, fmt.Errorf("recheck retrieved documents: %w", err)
	}
	defer rows.Close()
	ready := map[string]string{}
	for rows.Next() {
		var id, filename string
		if err := rows.Scan(&id, &filename); err != nil {
			return nil, err
		}
		ready[id] = filename
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}

	seen := map[string]bool{}
	accepted := make([]Candidate, 0, len(candidates))
	for _, candidate := range candidates {
		filename, ok := ready[candidate.DocumentID]
		if !ok || seen[candidate.CitationID] {
			continue
		}
		seen[candidate.CitationID] = true
		candidate.Filename = filename
		accepted = append(accepted, candidate)
	}
	// Filtering preserves Qdrant's sparse-score order; no fusion or local
	// ranking algorithm is applied here.
	return accepted, nil
}

func (q *qdrantIndex) SearchDense(ctx context.Context, scope retrievalScope, vector []float32, limit uint64) ([]string, error) {
	if limit == 0 {
		return []string{}, nil
	}
	threshold := scope.DenseThreshold
	points, err := q.client.Query(ctx, &qdrant.QueryPoints{
		CollectionName: q.denseCollection, Query: qdrant.NewQueryDense(vector),
		Filter: activeScopeFilter(scope), ScoreThreshold: &threshold, Limit: &limit,
		WithPayload: qdrant.NewWithPayloadInclude("document_id"),
	})
	if err != nil {
		return nil, err
	}
	ids := make([]string, 0, len(points))
	seen := map[string]bool{}
	for _, point := range points {
		id := payloadString(point.Payload, "document_id")
		if id != "" && !seen[id] {
			ids = append(ids, id)
			seen[id] = true
		}
	}
	return ids, nil
}

func (q *qdrantIndex) SearchSparse(ctx context.Context, scope retrievalScope, vector sparseVector, documentIDs []string, limit uint64) ([]Candidate, error) {
	if limit == 0 || len(vector.Indices) == 0 || len(vector.Indices) != len(vector.Values) {
		return []Candidate{}, nil
	}
	filter := activeScopeFilter(scope)
	if len(documentIDs) > 0 {
		filter.Must = append(filter.Must, qdrant.NewMatchKeywords("document_id", documentIDs...))
	}
	threshold, using := scope.SparseThreshold, "bm25"
	points, err := q.client.Query(ctx, &qdrant.QueryPoints{
		CollectionName: q.sparseCollection, Query: qdrant.NewQuerySparse(vector.Indices, vector.Values), Using: &using,
		Filter: filter, ScoreThreshold: &threshold, Limit: &limit, WithPayload: qdrant.NewWithPayload(true),
	})
	if err != nil {
		return nil, err
	}
	results := make([]Candidate, 0, len(points))
	for _, point := range points {
		documentID, text := payloadString(point.Payload, "document_id"), payloadString(point.Payload, "text")
		if documentID == "" || strings.TrimSpace(text) == "" {
			continue
		}
		chunkIndex := int(payloadInteger(point.Payload, "chunk_index"))
		citationID := point.Id.GetUuid()
		if citationID == "" {
			citationID = fmt.Sprintf("%s:%d", documentID, chunkIndex)
		}
		candidate := Candidate{Citation: Citation{CitationID: citationID, DocumentID: documentID, Filename: payloadString(point.Payload, "filename"), ChunkIndex: chunkIndex}, Text: text, Score: point.Score}
		if value, ok := point.Payload["page"]; ok {
			page := int(value.GetIntegerValue())
			candidate.Page = &page
		}
		results = append(results, candidate)
	}
	return results, nil
}

func activeScopeFilter(scope retrievalScope) *qdrant.Filter {
	return &qdrant.Filter{Must: []*qdrant.Condition{
		qdrant.NewMatchKeyword("namespace_id", scope.NamespaceID),
		qdrant.NewMatchKeyword("collection_id", scope.CollectionID),
		qdrant.NewMatchKeyword("index_version", scope.IndexVersion),
		qdrant.NewMatchBool("active", true),
	}}
}

func payloadString(payload map[string]*qdrant.Value, key string) string {
	if value, ok := payload[key]; ok {
		return value.GetStringValue()
	}
	return ""
}

func payloadInteger(payload map[string]*qdrant.Value, key string) int64 {
	if value, ok := payload[key]; ok {
		return value.GetIntegerValue()
	}
	return 0
}
