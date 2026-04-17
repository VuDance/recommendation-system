package com.vudance.recommendation_system.model;

import jakarta.persistence.*;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Positive;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;
import java.util.List;

import org.hibernate.annotations.JdbcTypeCode;
import org.hibernate.type.SqlTypes;

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
    
    @JdbcTypeCode(SqlTypes.JSON)
    @Column(name = "image_url", nullable = true, columnDefinition = "JSON")
    private List<String> imageURL;
}