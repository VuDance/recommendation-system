package com.vudance.flink.job.model;

import java.io.Serializable;
import java.util.Map;

/**
 * Intermediate POJO produced by {@code WindowedViewAggregator} and consumed
 * by {@link RecommendationEnrichmentOperator}.
 *
 * <p>Replaces the raw JSON string approach so Flink can handle the object
 * type-safely without extra deserialization in the enrichment stage.
 */
public class UserViewProfile implements Serializable {

    private static final long serialVersionUID = 1L;

    /** Keyed user identifier. */
    public final String userId;

    /** productId → number of views in this window. */
    public final Map<String, Long> viewedProducts;

    public final long windowStart;
    public final long windowEnd;

    public UserViewProfile(String userId, Map<String, Long> viewedProducts,
                           long windowStart, long windowEnd) {
        this.userId         = userId;
        this.viewedProducts = viewedProducts;
        this.windowStart    = windowStart;
        this.windowEnd      = windowEnd;
    }

    @Override
    public String toString() {
        return "UserViewProfile{userId='" + userId + '\'' +
               ", products=" + viewedProducts.size() +
               ", window=[" + windowStart + ',' + windowEnd + "]}";
    }
}