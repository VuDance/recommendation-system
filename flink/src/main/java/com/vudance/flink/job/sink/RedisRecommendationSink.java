package com.vudance.flink.job.sink;

import com.vudance.flink.job.service.MilvusVectorQueryService.ScoredProduct;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.streaming.api.functions.sink.RichSinkFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import redis.clients.jedis.Jedis;
import redis.clients.jedis.JedisPool;
import redis.clients.jedis.JedisPoolConfig;
import redis.clients.jedis.Pipeline;

import java.util.List;
import java.util.Map;

/**
 * Flink RichSinkFunction that writes per-user recommendation results to Redis.
 *
 * <p>Storage layout (Redis Hash):
 * <pre>
 *   Key:   recommendations:{userId}
 *   Field: {productId}
 *   Value: similarity score (float string)
 * </pre>
 *
 * <p>The hash is fully replaced on each window flush, then a TTL is applied
 * so stale recommendations expire automatically (default: 2 hours).
 */
public class RedisRecommendationSink extends RichSinkFunction<com.vudance.flink.job.sink.RedisRecommendationSink.UserRecommendation> {

    private static final long serialVersionUID = 1L;
    private static final Logger log = LoggerFactory.getLogger(RedisRecommendationSink.class);

    private static final String KEY_PREFIX = "recommendations:";
    private static final int    TTL_SECONDS = 60 * 60 * 2; // 2 hours

    private final String redisHost;
    private final int    redisPort;

    private transient JedisPool jedisPool;

    public RedisRecommendationSink(String redisHost, int redisPort) {
        this.redisHost = redisHost;
        this.redisPort = redisPort;
    }

    @Override
    public void open(Configuration parameters) {
        JedisPoolConfig poolConfig = new JedisPoolConfig();
        poolConfig.setMaxTotal(8);
        poolConfig.setMaxIdle(4);
        poolConfig.setMinIdle(1);
        poolConfig.setTestOnBorrow(true);

        jedisPool = new JedisPool(poolConfig, redisHost, redisPort);
        log.info("Redis connection pool initialized at {}:{}", redisHost, redisPort);
    }

    @Override
    public void invoke(UserRecommendation rec) throws Exception {
        if (rec.recommendations.isEmpty()) {
            log.debug("Empty recommendations for userId={}, skipping Redis write", rec.userId);
            return;
        }

        String redisKey = KEY_PREFIX + rec.userId;

        try (Jedis jedis = jedisPool.getResource(); Pipeline pipeline = jedis.pipelined()) {
            // Atomically delete old data and write new scores in one pipeline
            pipeline.del(redisKey);

            for (ScoredProduct sp : rec.recommendations) {
                pipeline.hset(redisKey, sp.productId, String.valueOf(sp.score));
            }

            // Set TTL so stale recs expire
            pipeline.expire(redisKey, TTL_SECONDS);

            pipeline.sync();

            log.debug("Wrote {} recommendations for userId={} (TTL={}s)",
                    rec.recommendations.size(), rec.userId, TTL_SECONDS);
        } catch (Exception e) {
            log.error("Failed to write recommendations to Redis for userId={}", rec.userId, e);
            // Flink will retry the checkpoint; do not rethrow to avoid task failure on transient errors
        }
    }

    @Override
    public void close() {
        if (jedisPool != null && !jedisPool.isClosed()) {
            jedisPool.close();
        }
    }

    // ── Value object passed through Flink ───────────────────────────────────────

    public static class UserRecommendation implements java.io.Serializable {
        private static final long serialVersionUID = 1L;

        public final String userId;
        public final List<ScoredProduct> recommendations;
        public final long windowStart;
        public final long windowEnd;

        public UserRecommendation(String userId, List<ScoredProduct> recommendations,
                                  long windowStart, long windowEnd) {
            this.userId          = userId;
            this.recommendations = recommendations;
            this.windowStart     = windowStart;
            this.windowEnd       = windowEnd;
        }
    }
}