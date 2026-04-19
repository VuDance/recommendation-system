package com.vudance.flink.util;

import org.apache.kafka.clients.admin.AdminClient;
import org.apache.kafka.clients.admin.AdminClientConfig;
import org.apache.kafka.clients.admin.CreateTopicsResult;
import org.apache.kafka.clients.admin.ListTopicsResult;
import org.apache.kafka.common.errors.TopicExistsException;
import org.apache.kafka.common.config.TopicConfig;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collections;
import java.util.Properties;
import java.util.Set;
import java.util.concurrent.ExecutionException;

/**
 * Utility class to initialize Kafka topics automatically.
 * Creates topics if they don't exist when the Flink job starts.
 */
public class KafkaTopicInitializer {

    private static final Logger log = LoggerFactory.getLogger(KafkaTopicInitializer.class);

    /**
     * Create a Kafka topic if it doesn't already exist.
     *
     * @param bootstrapServers Kafka bootstrap servers (e.g., "localhost:9092")
     * @param topicName Name of the topic to create
     * @param numPartitions Number of partitions for the topic
     * @param replicationFactor Replication factor for the topic
     */
    public static void createTopicIfNotExists(String bootstrapServers, String topicName,
                                              int numPartitions, short replicationFactor) {
        Properties adminProps = new Properties();
        adminProps.put(AdminClientConfig.BOOTSTRAP_SERVERS_CONFIG, bootstrapServers);

        try (AdminClient adminClient = AdminClient.create(adminProps)) {
            // Check if topic exists
            if (topicExists(adminClient, topicName)) {
                log.info("Topic '{}' already exists", topicName);
                return;
            }

            // Create topic
            log.info("Creating topic '{}' with {} partitions and replication factor {}",
                    topicName, numPartitions, replicationFactor);

            org.apache.kafka.common.config.ConfigResource configResource =
                    new org.apache.kafka.common.config.ConfigResource(
                            org.apache.kafka.common.config.ConfigResource.Type.TOPIC, topicName);

            org.apache.kafka.clients.admin.NewTopic newTopic =
                    new org.apache.kafka.clients.admin.NewTopic(topicName, numPartitions, replicationFactor);

            // Optional: Set default retention policy
            newTopic.configs(Collections.singletonMap(TopicConfig.RETENTION_MS_CONFIG, "604800000")); // 7 days

            CreateTopicsResult result = adminClient.createTopics(Collections.singleton(newTopic));

            try {
                result.all().get();
                log.info("Successfully created topic '{}'", topicName);
            } catch (ExecutionException e) {
                if (e.getCause() instanceof TopicExistsException) {
                    log.info("Topic '{}' already exists (race condition handled)", topicName);
                } else {
                    log.error("Failed to create topic '{}'", topicName, e);
                    throw e;
                }
            }
        } catch (Exception e) {
            log.error("Error initializing Kafka topic '{}'", topicName, e);
            throw new RuntimeException("Failed to initialize Kafka topic: " + topicName, e);
        }
    }

    /**
     * Check if a topic exists in the Kafka cluster.
     */
    private static boolean topicExists(AdminClient adminClient, String topicName) {
        try {
            ListTopicsResult topicsResult = adminClient.listTopics();
            Set<String> topics = topicsResult.names().get();
            return topics.contains(topicName);
        } catch (InterruptedException | ExecutionException e) {
            log.warn("Error checking if topic '{}' exists", topicName, e);
            return false;
        }
    }

    /**
     * Create multiple topics.
     *
     * @param bootstrapServers Kafka bootstrap servers
     * @param topics Map of topic names to partition count (replication factor defaults to 1)
     */
    public static void createTopicsIfNotExist(String bootstrapServers, java.util.Map<String, Integer> topics) {
        for (String topic : topics.keySet()) {
            createTopicIfNotExists(bootstrapServers, topic, topics.get(topic), (short) 1);
        }
    }
}
