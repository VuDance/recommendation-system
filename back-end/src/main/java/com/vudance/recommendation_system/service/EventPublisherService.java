package com.vudance.recommendation_system.service;

import com.vudance.recommendation_system.dto.UserViewEvent;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.stereotype.Service;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@Service
public class EventPublisherService {
    
    private static final Logger logger = LoggerFactory.getLogger(EventPublisherService.class);
    
    @Value("${kafka.topic.user-view-events:user-view-events}")
    private String userViewEventsTopic;
    
    private final KafkaTemplate<String, Object> kafkaTemplate;
    
    public EventPublisherService(KafkaTemplate<String, Object> kafkaTemplate) {
        this.kafkaTemplate = kafkaTemplate;
    }
    
    /**
     * Publish a user view event to Kafka
     * @param userId User ID (can be null for anonymous users)
     * @param productId Product ID
     */
    public void publishUserViewEvent(String userId, String productId) {
        try {
            UserViewEvent event = new UserViewEvent(
                    userId,
                    productId,
                    System.currentTimeMillis()
            );
            
            kafkaTemplate.send(userViewEventsTopic, productId, event);
            logger.info("Published user view event - UserId: {}, ProductId: {}", userId, productId);
        } catch (Exception e) {
            logger.error("Error publishing user view event", e);
            // Don't fail the product view endpoint if Kafka is down
        }
    }
}
