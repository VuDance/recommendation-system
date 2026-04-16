package com.vudance.recommendation_system.dto;

import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class RegisterRequest {
    
    @Email
    @NotBlank
    private String email;
    
    @NotBlank
    @Size(min = 6, message = "Password must be at least 6 characters")
    private String password;
    
    @NotBlank
    @Size(min = 2, max = 50)
    private String firstName;
    
    @NotBlank
    @Size(min = 2, max = 50)
    private String lastName;
    
    @Size(max = 200)
    private String address;
}
