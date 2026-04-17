package com.vudance.flink.job.service;

import io.milvus.client.MilvusServiceClient;
import io.milvus.grpc.QueryResults;
import io.milvus.grpc.SearchResults;
import io.milvus.param.ConnectParam;
import io.milvus.param.R;
import io.milvus.param.dml.QueryParam;
import io.milvus.param.dml.SearchParam;
import io.milvus.response.QueryResultsWrapper;
import io.milvus.response.SearchResultsWrapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Serializable;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Queries Milvus for similar products using a weighted average vector
 * derived from the user's view history (more views = higher weight).
 *
 * <p>Assumes a Milvus collection named "product_vectors" with:
 * <ul>
 *   <li>{@code product_id}  — VARCHAR primary key</li>
 *   <li>{@code embedding}   — FLOAT_VECTOR(N) field</li>
 * </ul>
 */
public class MilvusVectorQueryService implements Serializable, AutoCloseable {

    private static final long serialVersionUID = 1L;
    private static final Logger log = LoggerFactory.getLogger(MilvusVectorQueryService.class);

    private static final String COLLECTION_NAME = "content_based_products";
    private static final String VECTOR_FIELD    = "vector";
    private static final String ID_FIELD        = "product_id";
    private static final int    VECTOR_DIM      = 128; // adjust to your model's output dim
    private static final int    TOP_K           = 20;  // candidates before dedup

    private final String milvusHost;
    private final int    milvusPort;

    private transient MilvusServiceClient client;

    public MilvusVectorQueryService(String milvusHost, int milvusPort) {
        this.milvusHost = milvusHost;
        this.milvusPort = milvusPort;
    }

    /** Lazy-init: safe to call multiple times. */
    public void ensureConnected() {
        if (client == null) {
            client = new MilvusServiceClient(
                    ConnectParam.newBuilder()
                            .withHost(milvusHost)
                            .withPort(milvusPort)
                            .withAuthorization("root", "Milvus")
                            .build()
            );
            log.info("Connected to Milvus at {}:{}", milvusHost, milvusPort);
        }
    }

    /**
     * Builds a weighted-average query vector from the user's viewed products
     * and returns the top-K most similar product IDs with their scores.
     *
     * @param viewedProducts  map of productId → viewCount
     * @param vectorStore     in-process cache of productId → embedding (pre-loaded or fetched)
     * @param excludeProducts product IDs already seen by the user (to filter out)
     * @return ordered list of (productId, similarityScore) pairs
     */
    public List<ScoredProduct> querySimilarProducts(
            Map<String, Long> viewedProducts,
            Map<String, float[]> vectorStore,
            Set<String> excludeProducts
    ) {
        ensureConnected();
        Map<String, float[]> vtStore = fetchVectorsFromMilvus(viewedProducts.keySet());
        float[] queryVector = buildWeightedAverageVector(viewedProducts, vtStore);
        if (queryVector == null) {
            log.warn("Could not build query vector — no embeddings found for viewed products");
            return Collections.emptyList();
        }

        // Milvus expects List<List<Float>>
        List<Float> queryVectorList = new ArrayList<>(queryVector.length);
        for (float v : queryVector) queryVectorList.add(v);

        SearchParam searchParam = SearchParam.newBuilder()
                .withCollectionName(COLLECTION_NAME)
                .withVectorFieldName(VECTOR_FIELD)
                .withVectors(Collections.singletonList(queryVectorList))
                .withTopK(TOP_K + excludeProducts.size()) // fetch extra to allow dedup
                .withMetricType(io.milvus.param.MetricType.IP) // Inner Product (cosine after normalization)
                .withOutFields(Collections.singletonList(ID_FIELD))
                .build();

        R<SearchResults> response = client.search(searchParam);
        if (response.getStatus() != R.Status.Success.getCode()) {
            log.error("Milvus search failed: {}", response.getMessage());
            return Collections.emptyList();
        }

        SearchResultsWrapper wrapper = new SearchResultsWrapper(response.getData().getResults());
        List<SearchResultsWrapper.IDScore> idScores = wrapper.getIDScore(0); // query index 0

        List<ScoredProduct> results = new ArrayList<>();
        for (SearchResultsWrapper.IDScore idScore : idScores) {
            String productId = idScore.getFieldValues().get("product_id").toString();
            if (!excludeProducts.contains(productId)) {
                results.add(new ScoredProduct(productId, idScore.getScore()));
            }
            if (results.size() >= TOP_K) break;
        }

        log.debug("Milvus returned {} candidates, {} after dedup", idScores.size(), results.size());
        return results;
    }

    private Map<String, float[]> fetchVectorsFromMilvus(Set<String> productIds) {
        Map<String, float[]> vectorStore = new HashMap<>();
        
        List<String> outputFields = Arrays.asList("product_id", VECTOR_FIELD);
        String expr = "product_id in [" + String.join(",", productIds.stream()
                .map(id -> "'" + id + "'")
                .collect(Collectors.toList())) + "]";

        QueryParam queryParam = QueryParam.newBuilder()
                .withCollectionName(COLLECTION_NAME)
                .withExpr(expr)
                .withOutFields(outputFields)
                .build();

        R<QueryResults> response = client.query(queryParam);
        
        if (response.getStatus() == R.Status.Success.getCode()) {
            QueryResultsWrapper wrapper = new QueryResultsWrapper(response.getData());
            // Giả sử product_id là String và embedding là List<Float>
            for (QueryResultsWrapper.RowRecord record : wrapper.getRowRecords()) {
                String pid = (String) record.get("product_id");
                List<Float> fieldVector = (List<Float>) record.get(VECTOR_FIELD);
                
                // Chuyển List<Float> sang float[]
                float[] vector = new float[fieldVector.size()];
                for (int i = 0; i < fieldVector.size(); i++) {
                    vector[i] = fieldVector.get(i);
                }
                vectorStore.put(pid, vector);
            }
        } else {
            log.error("Failed to query Milvus: {}", response.getMessage());
        }
        return vectorStore;
    }
    
    public Map<String, float[]> fetchVectorsByIds(List<String> productIds) {
        ensureConnected();

        if (productIds.isEmpty()) {
            return Collections.emptyMap();
        }

        // Build expr: product_id in ["id1", "id2", ...]
        StringBuilder expr = new StringBuilder(ID_FIELD + " in [");
        for (int i = 0; i < productIds.size(); i++) {
            if (i > 0) expr.append(", ");
            expr.append("\"").append(productIds.get(i)).append("\"");
        }
        expr.append("]");

        io.milvus.param.dml.QueryParam queryParam = io.milvus.param.dml.QueryParam.newBuilder()
                .withCollectionName(COLLECTION_NAME)
                .withExpr(expr.toString())
                .withOutFields(Arrays.asList(ID_FIELD, VECTOR_FIELD))
                .build();

        R<io.milvus.grpc.QueryResults> response = client.query(queryParam);
        if (response.getStatus() != R.Status.Success.getCode()) {
            log.error("Milvus query failed: {}", response.getMessage());
            return Collections.emptyMap();
        }

        io.milvus.response.QueryResultsWrapper wrapper = new io.milvus.response.QueryResultsWrapper(response.getData());
        List<io.milvus.response.QueryResultsWrapper.RowRecord> records = wrapper.getRowRecords();

        Map<String, float[]> result = new HashMap<>();
        for (io.milvus.response.QueryResultsWrapper.RowRecord record : records) {
            String productId = (String) record.get(ID_FIELD);
            List<Float> vectorList = (List<Float>) record.get(VECTOR_FIELD);
            float[] vector = new float[vectorList.size()];
            for (int i = 0; i < vectorList.size(); i++) {
                vector[i] = vectorList.get(i);
            }
            result.put(productId, vector);
        }

        log.debug("Fetched {} vectors from Milvus", result.size());
        return result;
    }

    /**
     * Computes a weighted average of product vectors.
     * Products viewed more often contribute proportionally more.
     */
    private float[] buildWeightedAverageVector(
            Map<String, Long> viewedProducts,
            Map<String, float[]> vectorStore
    ) {
        float[] result = null;
        long totalWeight = 0;

        for (Map.Entry<String, Long> entry : viewedProducts.entrySet()) {
            float[] vec = vectorStore.get(entry.getKey());
            if (vec == null) {
                log.debug("No vector found for productId={}", entry.getKey());
                continue;
            }
            long weight = entry.getValue();
            totalWeight += weight;

            if (result == null) {
                result = new float[vec.length];
            }
            for (int i = 0; i < vec.length; i++) {
                result[i] += vec[i] * weight;
            }
        }

        if (result == null || totalWeight == 0) return null;

        // Normalize by total weight
        for (int i = 0; i < result.length; i++) {
            result[i] /= totalWeight;
        }

        // L2-normalize so Inner Product ≈ cosine similarity
        l2Normalize(result);
        return result;
    }

    private void l2Normalize(float[] vec) {
        double norm = 0;
        for (float v : vec) norm += (double) v * v;
        norm = Math.sqrt(norm);
        if (norm > 1e-9) {
            for (int i = 0; i < vec.length; i++) vec[i] /= (float) norm;
        }
    }

    @Override
    public void close() {
        if (client != null) {
            client.close();
            client = null;
        }
    }

    // ── Value object ────────────────────────────────────────────────────────────

    public static class ScoredProduct implements Serializable {
        private static final long serialVersionUID = 1L;

        public final String productId;
        public final float  score;

        public ScoredProduct(String productId, float score) {
            this.productId = productId;
            this.score     = score;
        }

        @Override
        public String toString() {
            return "ScoredProduct{productId='" + productId + "', score=" + score + '}';
        }
    }
}