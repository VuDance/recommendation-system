package com.vudance.recommendation_system.dto;

import jakarta.validation.constraints.NotBlank;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class ProductDTO {
    
    private String asin;
    
    @NotBlank
    private String title;
    
    private String description;
    
    private String brand;
    private String imageURL;
}