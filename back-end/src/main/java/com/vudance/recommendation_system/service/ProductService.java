package com.vudance.recommendation_system.service;

import com.vudance.recommendation_system.model.Product;
import com.vudance.recommendation_system.repository.ProductRepository;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;

@Service
public class ProductService {
    
    private final ProductRepository productRepository;
    
    public ProductService(ProductRepository productRepository) {
        this.productRepository = productRepository;
    }
    
    public List<Product> findAll() {
        return productRepository.findAll();
    }
    
    public Page<Product> findAll(Pageable pageable) {
        return productRepository.findAll(pageable);
    }
    
    public Optional<Product> findById(Integer id) {
        return productRepository.findById(id);
    }

    public List<Product> findAll(List<String> ids) {
        return productRepository.findAllByAsinIn(ids);
    }
    
    public List<Product> findByCategory(String brand) {
        return productRepository.findByBrand(brand);
    }
    
    public List<Product> searchProducts(String keyword) {
        return productRepository.findByKeyword(keyword);
    }
    
    public Product save(Product product) {
        return productRepository.save(product);
    }
    
    public void deleteById(Integer id) {
        productRepository.deleteById(id);
    }
    
    public boolean existsByName(String title) {
        return productRepository.existsByTitle(title);
    }
}