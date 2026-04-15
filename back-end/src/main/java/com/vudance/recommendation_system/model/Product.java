package com.vudance.recommendation_system.model;

import jakarta.persistence.*;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Positive;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@Entity
@Table(name = "products")
@Data
@NoArgsConstructor
@AllArgsConstructor
public class Product {
    
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Integer id;

    private String asin;

    private Integer train_id;
    
    @NotBlank
    @Column(nullable = false, columnDefinition = "TEXT")
    private String title;
    
    @Column(nullable = true, columnDefinition = "TEXT")
    private String description;
    
    @Column(name = "brand")
    private String brand;
    
    @Column(name = "image_url", columnDefinition = "TEXT")
    private String imageURL;
}