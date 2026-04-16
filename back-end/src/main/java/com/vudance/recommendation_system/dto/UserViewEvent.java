package com.vudance.recommendation_system.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class UserViewEvent {
    private String userId;
    private String productId;
    private long timestamp;
}
