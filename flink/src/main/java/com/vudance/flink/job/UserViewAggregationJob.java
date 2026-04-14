package com.vudance.flink.job;

import com.vudance.flink.job.model.UserViewEvent;
import com.vudance.flink.job.operator.RecommendationEnrichmentOperator;
import com.vudance.flink.job.model.UserViewProfile;
import com.vudance.flink.job.serialization.UserViewEventDeserializationSchema;
import com.vudance.flink.job.sink.RedisRecommendationSink;
import com.vudance.flink.job.sink.RedisRecommendationSink.UserRecommendation;
import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.api.common.state.ValueState;
import org.apache.flink.api.common.state.ValueStateDescriptor;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.connector.kafka.source.KafkaSource;
import org.apache.flink.connector.kafka.source.enumerator.initializer.OffsetsInitializer;
import org.apache.flink.streaming.api.datastream.AsyncDataStream;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.KeyedProcessFunction;
import org.apache.flink.streaming.api.windowing.time.Time;
import org.apache.flink.util.Collector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/**
 * Extended Flink job pipeline:
 *
 * <pre>
 *  Kafka (user-view-events)
 *       │
 *       ▼
 *  keyBy(userId)
 *       │
 *       ▼
 *  WindowedViewAggregator          ← tumbling processing-time window (1 min)
 *  emits UserViewProfile
 *       │
 *       ▼
 *  AsyncDataStream                 ← non-blocking, 4 concurrent Milvus queries
 *  RecommendationEnrichmentOperator
 *       │  weighted-avg vector → Milvus ANN search → top-20 products
 *       ▼
 *  RedisRecommendationSink         ← HSET recommendations:{userId} productId score
 *                                     + EXPIRE 7200
 * </pre>
 */
public class UserViewAggregationJob {

    private static final Logger log = LoggerFactory.getLogger(UserViewAggregationJob.class);

    // ── Infrastructure addresses ─────────────────────────────────────────────
    private static final String KAFKA_BOOTSTRAP_SERVERS = "localhost:9092";
    private static final String MILVUS_HOST             = "localhost";
    private static final int    MILVUS_PORT             = 19530;
    private static final String REDIS_HOST              = "localhost";
    private static final int    REDIS_PORT              = 6379;

    // ── Topic names ───────────────────────────────────────────────────────────
    private static final String INPUT_TOPIC  = "user-view-events";

    // ── Window ────────────────────────────────────────────────────────────────
    private static final long WINDOW_SIZE_MINUTES = 1;

    // ── Async operator config ─────────────────────────────────────────────────
    private static final int  ASYNC_CAPACITY        = 100; // max in-flight async requests
    private static final long ASYNC_TIMEOUT_SECONDS = 10;

    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.enableCheckpointing(5_000L);
        env.setParallelism(2);

        // ── Source ───────────────────────────────────────────────────────────
        KafkaSource<UserViewEvent> source = KafkaSource.<UserViewEvent>builder()
                .setBootstrapServers(KAFKA_BOOTSTRAP_SERVERS)
                .setTopics(INPUT_TOPIC)
                .setGroupId("flink-view-aggregator")
                .setStartingOffsets(OffsetsInitializer.earliest())
                .setValueOnlyDeserializer(new UserViewEventDeserializationSchema())
                .build();

        DataStream<UserViewEvent> viewStream = env
                .fromSource(source, WatermarkStrategy.noWatermarks(), "kafka-source")
                .name("User View Source");
        viewStream.print("Raw Event");
        // ── Aggregate view counts per user per window ─────────────────────────
        DataStream<com.vudance.flink.job.model.UserViewProfile> profiles = viewStream
                .keyBy(UserViewEvent::getUserId)
                .process(new WindowedViewAggregator())
                .name("User View Aggregator");

        // ── Async enrich: Milvus vector search ────────────────────────────────
        DataStream<UserRecommendation> recommendations = AsyncDataStream.unorderedWait(
                profiles,
                new RecommendationEnrichmentOperator(MILVUS_HOST, MILVUS_PORT),
                ASYNC_TIMEOUT_SECONDS,
                TimeUnit.SECONDS,
                ASYNC_CAPACITY
        ).name("Milvus Recommendation Enrichment");

        // ── Sink: Redis ───────────────────────────────────────────────────────
        recommendations
                .addSink(new RedisRecommendationSink(REDIS_HOST, REDIS_PORT))
                .name("Redis Recommendation Sink");

        // Debug
        recommendations
                .map(r -> String.format("userId=%s recs=%d", r.userId, r.recommendations.size()))
                .print()
                .name("Debug Console");

        env.execute("User View Aggregation + Recommendation Job");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Aggregator: same logic as before, but emits UserViewProfile (typed POJO)
    // instead of raw JSON string.
    // ─────────────────────────────────────────────────────────────────────────

    public static class WindowedViewAggregator
            extends KeyedProcessFunction<String, UserViewEvent, UserViewProfile> {

        private static final long serialVersionUID = 1L;

        private transient ValueState<Map<String, Long>> viewCounts;
        private transient ValueState<Long> windowEnd;

        @Override
        public void open(Configuration parameters) {
            viewCounts = getRuntimeContext().getState(
                    new ValueStateDescriptor<>("view-counts",
                            Types.MAP(Types.STRING, Types.LONG))
            );
            windowEnd = getRuntimeContext().getState(
                    new ValueStateDescriptor<>("window-end", Types.LONG)
            );
        }

        @Override
        public void processElement(UserViewEvent event, Context ctx,
                                   Collector<UserViewProfile> out) throws Exception {
            long now = ctx.timerService().currentProcessingTime();

            Long currentWindowEnd = windowEnd.value();
            if (currentWindowEnd == null) {
                long end = now + Time.minutes(WINDOW_SIZE_MINUTES).toMilliseconds();
                windowEnd.update(end);
                ctx.timerService().registerProcessingTimeTimer(end);
            }

            Map<String, Long> counts = viewCounts.value();
            if (counts == null) counts = new HashMap<>();
            counts.merge(event.getProductId(), 1L, Long::sum);
            viewCounts.update(counts);
        }

        @Override
        public void onTimer(long timestamp, OnTimerContext ctx,
                            Collector<UserViewProfile> out) throws Exception {
            Map<String, Long> counts = viewCounts.value();

            if (counts != null && !counts.isEmpty()) {
                long start = timestamp - Time.minutes(WINDOW_SIZE_MINUTES).toMilliseconds();
                out.collect(new UserViewProfile(ctx.getCurrentKey(), counts, start, timestamp));
                viewCounts.clear();
            }

            // Always schedule next window (even if empty, to keep timer alive)
            long nextEnd = timestamp + Time.minutes(WINDOW_SIZE_MINUTES).toMilliseconds();
            windowEnd.update(nextEnd);
            ctx.timerService().registerProcessingTimeTimer(nextEnd);
        }
    }
}