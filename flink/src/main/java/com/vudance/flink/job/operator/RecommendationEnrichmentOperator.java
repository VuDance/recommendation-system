package com.vudance.flink.job.operator;

import com.vudance.flink.job.model.UserViewProfile;
import com.vudance.flink.job.service.MilvusVectorQueryService;
import com.vudance.flink.job.service.MilvusVectorQueryService.ScoredProduct;
import com.vudance.flink.job.sink.RedisRecommendationSink.UserRecommendation;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.streaming.api.functions.async.ResultFuture;
import org.apache.flink.streaming.api.functions.async.RichAsyncFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.concurrent.*;

/**
 * Async Flink operator that:
 *
 * <ol>
 *   <li>Receives an aggregated {@link UserViewProfile} from {@code WindowedViewAggregator}</li>
 *   <li>Fetches product embeddings from an in-memory cache (populated via Milvus or a local store)</li>
 *   <li>Queries Milvus for similar products using weighted-average vector</li>
 *   <li>Emits a {@link UserRecommendation} for the Redis sink</li>
 * </ol>
 *
 * <p>Uses a thread-pool executor so Milvus I/O does not block the Flink task thread.
 */
public class RecommendationEnrichmentOperator
        extends RichAsyncFunction<UserViewProfile, UserRecommendation> {

    private static final long serialVersionUID = 1L;
    private static final Logger log = LoggerFactory.getLogger(RecommendationEnrichmentOperator.class);

    private final String milvusHost;
    private final int    milvusPort;

    // In production, replace with a distributed cache (e.g., Redis or a pre-loaded broadcast state)
    // Here we use a simple ConcurrentHashMap populated lazily from Milvus "get" calls.
    private transient Map<String, float[]> vectorCache;
    private transient MilvusVectorQueryService milvusService;
    private transient ExecutorService executor;

    public RecommendationEnrichmentOperator(String milvusHost, int milvusPort) {
        this.milvusHost = milvusHost;
        this.milvusPort = milvusPort;
    }

    @Override
    public void open(Configuration parameters) {
        milvusService = new MilvusVectorQueryService(milvusHost, milvusPort);
        milvusService.ensureConnected();

        vectorCache = new ConcurrentHashMap<>();
        // Thread pool: number of concurrent Milvus queries in flight
        executor = Executors.newFixedThreadPool(4);
        log.info("RecommendationEnrichmentOperator opened");
    }

    @Override
    public void asyncInvoke(UserViewProfile profile, ResultFuture<UserRecommendation> resultFuture) throws Exception {
        executor.submit(() -> {
            try {
                // 1. Ensure embeddings are available for all viewed products
                loadMissingVectors(new HashSet<>(profile.viewedProducts.keySet()));

                // 2. Query Milvus for top-K similar products
                Set<String> alreadySeen = profile.viewedProducts.keySet();
                List<ScoredProduct> recommendations = milvusService.querySimilarProducts(
                        profile.viewedProducts,
                        vectorCache,
                        alreadySeen
                );

                // 3. Emit result
                resultFuture.complete(Collections.singletonList(
                        new UserRecommendation(
                                profile.userId,
                                recommendations,
                                profile.windowStart,
                                profile.windowEnd
                        )
                ));
            } catch (Exception e) {
                log.error("Error enriching recommendations for userId={}", profile.userId, e);
                resultFuture.completeExceptionally(e);
            }
        });
    }

    @Override
    public void timeout(UserViewProfile input, ResultFuture<UserRecommendation> resultFuture) {
        log.warn("Milvus query timed out for userId={}", input.userId);
        // Emit empty recommendation instead of dropping the event
        resultFuture.complete(Collections.singletonList(
                new UserRecommendation(input.userId, Collections.emptyList(),
                        input.windowStart, input.windowEnd)
        ));
    }

    @Override
    public void close() throws Exception {
        if (executor != null) executor.shutdown();
        if (milvusService != null) milvusService.close();
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    /**
     * Fetches vectors for product IDs not yet in the local cache.
     * Uses Milvus "query" (not "search") to retrieve embeddings by ID.
     *
     * In production, consider using a broadcast state populated from a batch job
     * to avoid per-event Milvus round-trips.
     */
    private void loadMissingVectors(Set<String> productIds) {
        List<String> missing = new ArrayList<>();
        for (String id : productIds) {
            if (!vectorCache.containsKey(id)) missing.add(id);
        }
        if (missing.isEmpty()) return;

        try {
            Map<String, float[]> fetched = milvusService.fetchVectorsByIds(missing);
            vectorCache.putAll(fetched);
            log.debug("Loaded {} new vectors into cache", fetched.size());
        } catch (Exception e) {
            log.warn("Failed to load vectors for {} products: {}", missing.size(), e.getMessage());
        }
    }
}