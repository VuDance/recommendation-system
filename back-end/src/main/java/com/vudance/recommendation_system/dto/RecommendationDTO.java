package com.vudance.recommendation_system.dto;

import java.util.Map;

/**
 * Data Transfer Object for recommendation response
 */
public class RecommendationDTO {

    private String userId;
    private Map<String, Integer> viewedProducts;
    private long windowStart;
    private long windowEnd;

    public String getUserId() {
        return userId;
    }

    public void setUserId(String userId) {
        this.userId = userId;
    }

    public Map<String, Integer> getViewedProducts() {
        return viewedProducts;
    }

    public void setViewedProducts(Map<String, Integer> viewedProducts) {
        this.viewedProducts = viewedProducts;
    }

    public long getWindowStart() {
        return windowStart;
    }

    public void setWindowStart(long windowStart) {
        this.windowStart = windowStart;
    }

    public long getWindowEnd() {
        return windowEnd;
    }

    public void setWindowEnd(long windowEnd) {
        this.windowEnd = windowEnd;
    }
}