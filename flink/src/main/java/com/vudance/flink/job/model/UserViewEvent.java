package com.vudance.flink.job.model;

import java.io.Serializable;

/**
 * Represents a user view event produced by the Java backend whenever a user views a product.
 * Sent to Kafka topic {@code user-view-events}.
 *
 * JSON format: {"userId": "user-123", "productId": "B00ABC123", "timestamp": 1712000000000}
 */
public class UserViewEvent implements Serializable {
    private static final long serialVersionUID = 1L;

    private String userId;
    private String productId;
    private long timestamp;

    public UserViewEvent() {}

    public UserViewEvent(String userId, String productId, long timestamp) {
        this.userId = userId;
        this.productId = productId;
        this.timestamp = timestamp;
    }

    public String getUserId() { return userId; }
    public void setUserId(String userId) { this.userId = userId; }

    public String getProductId() { return productId; }
    public void setProductId(String productId) { this.productId = productId; }

    public long getTimestamp() { return timestamp; }
    public void setTimestamp(long timestamp) { this.timestamp = timestamp; }

    @Override
    public String toString() {
        return "UserViewEvent{userId='" + userId + "', productId='" + productId + "', timestamp=" + timestamp + "}";
    }
}
