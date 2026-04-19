package com.vudance.flink.config;

/**
 * Configuration management for Flink jobs.
 * Loads configuration from environment variables with sensible defaults.
 * 
 * Usage:
 * <pre>
 *     FlinkJobConfig config = FlinkJobConfig.getInstance();
 *     String kafkaServers = config.getKafkaBootstrapServers();
 * </pre>
 */
public class FlinkJobConfig {
    
    private static FlinkJobConfig instance;

    // Kafka Configuration
    private final String kafkaBootstrapServers;
    private final String kafkaTopicUserViewEvents;

    // Milvus Configuration
    private final String milvusHost;
    private final int milvusPort;

    // Redis Configuration
    private final String redisHost;
    private final int redisPort;

    // Flink Configuration
    private final long windowSizeMinutes;
    private final int asyncCapacity;
    private final long asyncTimeoutSeconds;

    private FlinkJobConfig() {
        // Kafka
        this.kafkaBootstrapServers = getEnv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092");
        this.kafkaTopicUserViewEvents = getEnv("KAFKA_TOPIC_USER_VIEW_EVENTS", "user-view-events");

        // Milvus
        this.milvusHost = getEnv("MILVUS_HOST", "localhost");
        this.milvusPort = getEnvInt("MILVUS_PORT", 19530);

        // Redis
        this.redisHost = getEnv("REDIS_HOST", "localhost");
        this.redisPort = getEnvInt("REDIS_PORT", 6379);

        // Flink
        this.windowSizeMinutes = getEnvLong("FLINK_WINDOW_SIZE_MINUTES", 1);
        this.asyncCapacity = getEnvInt("FLINK_ASYNC_CAPACITY", 100);
        this.asyncTimeoutSeconds = getEnvLong("FLINK_ASYNC_TIMEOUT_SECONDS", 10);
    }

    /**
     * Get singleton instance of configuration.
     */
    public static synchronized FlinkJobConfig getInstance() {
        if (instance == null) {
            instance = new FlinkJobConfig();
        }
        return instance;
    }

    /**
     * Get environment variable or return default value.
     */
    private static String getEnv(String key, String defaultValue) {
        String value = System.getenv(key);
        return value != null && !value.isEmpty() ? value : defaultValue;
    }

    /**
     * Get environment variable as integer or return default value.
     */
    private static int getEnvInt(String key, int defaultValue) {
        try {
            String value = System.getenv(key);
            return value != null && !value.isEmpty() ? Integer.parseInt(value) : defaultValue;
        } catch (NumberFormatException e) {
            System.err.println("Failed to parse environment variable " + key + " as integer, using default: " + defaultValue);
            return defaultValue;
        }
    }

    /**
     * Get environment variable as long or return default value.
     */
    private static long getEnvLong(String key, long defaultValue) {
        try {
            String value = System.getenv(key);
            return value != null && !value.isEmpty() ? Long.parseLong(value) : defaultValue;
        } catch (NumberFormatException e) {
            System.err.println("Failed to parse environment variable " + key + " as long, using default: " + defaultValue);
            return defaultValue;
        }
    }

    // Getters
    public String getKafkaBootstrapServers() {
        return kafkaBootstrapServers;
    }

    public String getKafkaTopicUserViewEvents() {
        return kafkaTopicUserViewEvents;
    }

    public String getMilvusHost() {
        return milvusHost;
    }

    public int getMilvusPort() {
        return milvusPort;
    }

    public String getRedisHost() {
        return redisHost;
    }

    public int getRedisPort() {
        return redisPort;
    }

    public long getWindowSizeMinutes() {
        return windowSizeMinutes;
    }

    public int getAsyncCapacity() {
        return asyncCapacity;
    }

    public long getAsyncTimeoutSeconds() {
        return asyncTimeoutSeconds;
    }
}
