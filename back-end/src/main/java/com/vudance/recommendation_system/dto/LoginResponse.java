package com.vudance.recommendation_system.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class LoginResponse {
    
    private Long userId;
    private String email;
    private String firstName;
    private String lastName;
    private String token;
}
