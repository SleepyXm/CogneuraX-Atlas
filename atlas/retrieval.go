package atlas

import (
	"context"
	"fmt"
	"regexp"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/lib/pq"
	"github.com/qdrant/go-client/qdrant"
)

func (p *processorClient) SparseQuery(ctx context.Context, query string) (sparseVector, error) {
	var vector sparseVector
	err := p.postJSON(ctx, "/sparse-query", map[string]string{"query": query}, &vector)
	return vector, err
}

func (s *Service) retrieveEvidenceRegions(ctx context.Context, namespaceID, collectionID, question string) ([]Candidate, error) {
	// 1. Authorize through PostgreSQL and size searches from ready content.
	var indexVersion string
	var readyRegions int
	err := s.db.QueryRowContext(ctx, `SELECT b.index_version,coalesce(sum(d.chunk_count),0) FROM atlas_collections b LEFT JOIN atlas_documents d ON d.collection_id=b.id AND d.status='ready' WHERE b.id=$1::uuid AND b.namespace_id=$2::uuid GROUP BY b.index_version`, collectionID, namespaceID).Scan(&indexVersion, &readyRegions)
	if err != nil {
		return nil, err
	}
	if indexVersion != s.indexVersion || readyRegions == 0 {
		return []Candidate{}, nil
	}

	// 2. Keep the grammatical question for BGE while separating lexical content
	// and exact identity spans. The pass copies source words; it invents nothing.
	pass := partitionPrompt(question)
	dense, err := s.embedder.Embed(ctx, []string{s.queryPrefix + pass.Dense})
	if err != nil {
		return nil, fmt.Errorf("embed retrieval question: %w", err)
	}
	if len(dense) != 1 {
		return nil, fmt.Errorf("TEI returned %d prompt embeddings, expected 1", len(dense))
	}
	scope := retrievalScope{NamespaceID: namespaceID, CollectionID: collectionID, IndexVersion: indexVersion, DenseThreshold: s.denseThreshold, SparseThreshold: s.sparseThreshold}
	// 3. Dense similarity establishes the semantic region set.
	regionIDs, err := s.index.SearchDense(ctx, scope, dense[0], uint64(readyRegions))
	if err != nil {
		return nil, fmt.Errorf("dense abstract gate: %w", err)
	}
	if len(regionIDs) == 0 {
		return []Candidate{}, nil
	}

	// 4. Only a non-empty community needs BM25 encoding. Exact names and numbers
	// must exist somewhere in that set, while the final evidence region may use
	// a reference or shortened name. There is no whole-collection fallback.
	sparse, err := s.processor.SparseQuery(ctx, pass.Sparse)
	if err != nil {
		return nil, fmt.Errorf("encode sparse retrieval question: %w", err)
	}
	candidates, err := s.index.SearchSparse(ctx, scope, sparse, regionIDs, pass.Anchors, uint64(len(regionIDs)))
	if err != nil {
		return nil, fmt.Errorf("sparse search inside dense evidence regions: %w", err)
	}
	// 5. Reject stale Qdrant points while preserving sparse-score order.
	return s.validateReadyCandidates(ctx, namespaceID, collectionID, indexVersion, candidates)
}

var promptWords = regexp.MustCompile(`[\p{L}\p{N}][\p{L}\p{N}'’.-]*`)
var sparsePromptNoise = map[string]bool{"who": true, "what": true, "when": true, "where": true, "why": true, "which": true, "how": true, "many": true, "much": true, "often": true, "according": true, "please": true, "tell": true, "show": true, "find": true, "give": true, "me": true, "the": true, "a": true, "an": true, "of": true, "to": true, "in": true, "on": true, "at": true, "by": true, "for": true, "from": true, "with": true, "and": true, "or": true, "is": true, "are": true, "was": true, "were": true, "do": true, "does": true, "did": true, "has": true, "have": true, "had": true}
var questionWords = map[string]bool{"who": true, "what": true, "when": true, "where": true, "why": true, "which": true, "how": true}

func partitionPrompt(question string) promptPass {
	// Dense keeps the full wording. Exact anchors and lexical terms are copied
	// from the prompt without model generation or inferred entities.
	words := promptWords.FindAllString(strings.TrimSpace(question), -1)
	if len(words) == 0 {
		return promptPass{Dense: question, Sparse: question}
	}
	exactAnchors, sparse := []string{}, []string{}
	for index := 0; index < len(words); {
		word, lower := words[index], strings.ToLower(words[index])
		if strings.IndexFunc(word, func(r rune) bool { return !unicode.IsDigit(r) }) == -1 {
			exactAnchors = append(exactAnchors, word)
			sparse = append(sparse, word)
			index++
			continue
		}
		first, _ := utf8.DecodeRuneInString(word)
		proper := index > 0 && !questionWords[lower] && unicode.IsUpper(first) && !strings.ContainsAny(word, ".")
		if proper {
			end := index + 1
			possessiveFirst := strings.HasSuffix(word, "'s") || strings.HasSuffix(word, "’s")
			for !possessiveFirst && end < len(words) {
				next, nextLower := words[end], strings.ToLower(words[end])
				r, _ := utf8.DecodeRuneInString(next)
				connector := end+1 < len(words) && (nextLower == "of" || nextLower == "the" || nextLower == "v" || nextLower == "vs")
				followedByProper := false
				if connector {
					afterConnector, _ := utf8.DecodeRuneInString(words[end+1])
					followedByProper = unicode.IsUpper(afterConnector)
				}
				if unicode.IsUpper(r) || followedByProper {
					end++
					continue
				}
				break
			}
			anchorWords := append([]string(nil), words[index:end]...)
			for i := range anchorWords {
				anchorWords[i] = strings.TrimSuffix(strings.TrimSuffix(anchorWords[i], "'s"), "’s")
			}
			anchor := strings.Join(anchorWords, " ")
			sparse = append(sparse, anchor)
			if end-index > 1 {
				exactAnchors = append(exactAnchors, anchor)
			}
			index = end
			continue
		}
		if !sparsePromptNoise[lower] && len([]rune(word)) > 1 {
			sparse = append(sparse, word)
		}
		index++
	}
	return promptPass{Dense: strings.TrimSpace(question), Sparse: strings.Join(sparse, " "), Anchors: exactAnchors}
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
		WithPayload: qdrant.NewWithPayloadInclude("region_id"),
	})
	if err != nil {
		return nil, err
	}
	ids := make([]string, 0, len(points))
	seen := map[string]bool{}
	for _, point := range points {
		id := payloadString(point.Payload, "region_id")
		if id != "" && !seen[id] {
			ids = append(ids, id)
			seen[id] = true
		}
	}
	return ids, nil
}

func (q *qdrantIndex) SearchSparse(ctx context.Context, scope retrievalScope, vector sparseVector, regionIDs, exactAnchors []string, limit uint64) ([]Candidate, error) {
	if limit == 0 || len(regionIDs) == 0 || len(vector.Indices) == 0 || len(vector.Indices) != len(vector.Values) {
		return []Candidate{}, nil
	}
	filter := activeScopeFilter(scope)
	filter.Must = append(filter.Must, qdrant.NewMatchKeywords("region_id", regionIDs...))
	// Anchors constrain the dense semantic set rather than every final region.
	for _, anchor := range exactAnchors {
		anchorFilter := activeScopeFilter(scope)
		anchorFilter.Must = append(anchorFilter.Must,
			qdrant.NewMatchKeywords("region_id", regionIDs...),
			qdrant.NewMatchPhrase("text", anchor),
		)
		exact := true
		count, err := q.client.Count(ctx, &qdrant.CountPoints{CollectionName: q.sparseCollection, Filter: anchorFilter, Exact: &exact})
		if err != nil {
			return nil, err
		}
		if count == 0 {
			return []Candidate{}, nil
		}
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
		chunkIndex, regionID := int(payloadInteger(point.Payload, "chunk_index")), payloadString(point.Payload, "region_id")
		if regionID == "" {
			continue
		}
		candidate := Candidate{Citation: Citation{CitationID: regionID, DocumentID: documentID, Filename: payloadString(point.Payload, "filename"), ChunkIndex: chunkIndex}, Text: text, Score: point.Score}
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
