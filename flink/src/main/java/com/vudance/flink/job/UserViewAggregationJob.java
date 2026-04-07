package com.vudance.flink.job;

import com.vudance.flink.job.model.UserViewEvent;
import com.vudance.flink.job.serialization.UserViewEventDeserializationSchema;
import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.serialization.SimpleStringSchema;
import org.apache.flink.api.common.state.ValueState;
import org.apache.flink.api.common.state.ValueStateDescriptor;
import org.apache.flink.connector.kafka.sink.KafkaRecordSerializationSchema;
import org.apache.flink.connector.kafka.sink.KafkaSink;
import org.apache.flink.connector.kafka.source.KafkaSource;
import org.apache.flink.connector.kafka.source.enumerator.initializer.OffsetsInitializer;
import org.apache.flink.shaded.jackson2.com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.KeyedProcessFunction;
import org.apache.flink.streaming.api.windowing.assigners.TumblingProcessingTimeWindows;
import org.apache.flink.streaming.api.windowing.time.Time;
import org.apache.flink.util.Collector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;

/**
 * Flink streaming job that consumes user product view events from Kafka,
 * aggregates counts per user over a processing-time window, and publishes
 * user interest profiles to an output Kafka topic for the ML service to consume.
 *
 * <p>Kafka topics:
 * <ul>
 *   <li>{@code user-view-events} (input) — published by Java backend on each product view</li>
 *   <li>{@code user-interest-profiles} (output) — aggregated profiles consumed by ML service</li>
 * </ul>
 *
 * <p>Input event JSON:
 * <pre>
 * {"userId": "user-123", "productId": "B00ABC123", "timestamp": 1712000000000}
 * </pre>
 *
 * <p>Output event JSON:
 * <pre>
 * {
 *   "userId": "user-123",
 *   "viewedProducts": {"B00ABC123": 3, "B00XYZ789": 1},
 *   "windowStart": 1712000000000,
 *   "windowEnd": 1712003600000
 * }
 * </pre>
 */
public class UserViewAggregationJob {
    private static final Logger log = LoggerFactory.getLogger(UserViewAggregationJob.class);

    // Kafka configuration — use Docker-internal address (containers on same network)
    private static final String KAFKA_BOOTSTRAP_SERVERS = "kafka:29092";
    private static final String INPUT_TOPIC = "user-view-events";
    private static final String OUTPUT_TOPIC = "user-interest-profiles";

    // Window configuration
    private static final long WINDOW_SIZE_MINUTES = 1;

    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.enableCheckpointing(5_000L);
        env.setParallelism(2);

        // ── Source: Kafka ──────────────────────────────────────────
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

        // ── Key by userId, process and aggregate ────────────────────
        DataStream<String> profiles = viewStream
                .keyBy(UserViewEvent::getUserId)
                .process(new WindowedViewAggregator())
                .name("User View Aggregator");

        // ── Sink: Kafka ─────────────────────────────────────────────
        KafkaSink<String> sink = KafkaSink.<String>builder()
                .setBootstrapServers(KAFKA_BOOTSTRAP_SERVERS)
                .setRecordSerializer(
                        KafkaRecordSerializationSchema.<String>builder()
                                .setTopic(OUTPUT_TOPIC)
                                .setValueSerializationSchema(new SimpleStringSchema())
                                .build()
                )
                .build();

        profiles.sinkTo(sink).name("User Interest Kafka Sink");

        // Also print to stdout for debugging
        profiles.print().name("Debug Console Print");

        env.execute("User View Aggregation Job");
    }

    /**
     * Aggregates view counts per user within a processing-time tumbling window (60 min).
     * Emits the aggregated result when the window fires.
     */
    public static class WindowedViewAggregator extends KeyedProcessFunction<String, UserViewEvent, String> {

        private static final long serialVersionUID = 1L;
        private final ObjectMapper mapper = new ObjectMapper();

        private transient ValueState<Map<String, Long>> viewCounts;
        private transient ValueState<Long> windowEnd;

        @Override
        public void open(org.apache.flink.configuration.Configuration parameters) {
            viewCounts = getRuntimeContext().getState(
                    new ValueStateDescriptor<>("view-counts", org.apache.flink.api.common.typeinfo.Types.MAP(
                            org.apache.flink.api.common.typeinfo.Types.STRING,
                            org.apache.flink.api.common.typeinfo.Types.LONG
                    ))
            );
            windowEnd = getRuntimeContext().getState(
                    new ValueStateDescriptor<>("window-end", org.apache.flink.api.common.typeinfo.Types.LONG)
            );
        }

        @Override
        public void processElement(UserViewEvent event, Context ctx, Collector<String> out) throws Exception {
            long currentTime = ctx.timerService().currentProcessingTime();

            // If this is the first event, set up the window timer
            Long currentWindowEnd = windowEnd.value();
            if (currentWindowEnd == null) {
                long windowEndTs = currentTime + (Time.minutes(WINDOW_SIZE_MINUTES).toMilliseconds());
                windowEnd.update(windowEndTs);
                ctx.timerService().registerProcessingTimeTimer(windowEndTs);
            }

            // Increment view count for the product
            Map<String, Long> counts = viewCounts.value();
            if (counts == null) {
                counts = new HashMap<>();
            }
            counts.merge(event.getProductId(), 1L, Long::sum);
            viewCounts.update(counts);
        }

        @Override
        public void onTimer(long timestamp, OnTimerContext ctx, Collector<String> out) throws Exception {
            Map<String, Long> counts = viewCounts.value();
            if (counts != null && !counts.isEmpty()) {
                Map<String, Object> profile = new HashMap<>();
                profile.put("userId", ctx.getCurrentKey());
                profile.put("viewedProducts", counts);
                // Compute approximate window bounds
                long startTime = timestamp - Time.minutes(WINDOW_SIZE_MINUTES).toMilliseconds();
                profile.put("windowStart", startTime);
                profile.put("windowEnd", timestamp);

                out.collect(mapper.writeValueAsString(profile));

                // Reset state and register next window
                viewCounts.clear();
                long nextWindowEnd = timestamp + Time.minutes(WINDOW_SIZE_MINUTES).toMilliseconds();
                windowEnd.update(nextWindowEnd);
                ctx.timerService().registerProcessingTimeTimer(nextWindowEnd);
            } else {
                windowEnd.clear();
            }
        }
    }
}
