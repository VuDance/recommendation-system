package com.vudance.recommendation_system.util;

import com.vudance.recommendation_system.dto.ProductDTO;
import com.vudance.recommendation_system.model.Product;
import org.springframework.stereotype.Component;

@Component
public class ModelMapper {
    
    // Product mapping methods
    public ProductDTO toProductDTO(Product product) {
        if (product == null) {
            return null;
        }
        
        ProductDTO dto = new ProductDTO();
        dto.setAsin(product.getAsin());
        dto.setTitle(product.getTitle());
        dto.setDescription(product.getDescription());
        dto.setBrand(product.getBrand());
        dto.setImageURL(product.getImageURL());
        
        return dto;
    }
    
    public Product toProductEntity(ProductDTO dto) {
        if (dto == null) {
            return null;
        }
        
        Product product = new Product();
        product.setAsin(dto.getAsin());
        product.setTitle(dto.getTitle());
        product.setDescription(dto.getDescription());
        product.setBrand(dto.getBrand());
        product.setImageURL(dto.getImageURL());
        
        return product;
    }
}