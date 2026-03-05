package com.vudance.recommendation_system.repository;

import com.vudance.recommendation_system.model.Product;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface ProductRepository extends JpaRepository<Product, String> {
    
    @Query("SELECT p FROM Product p WHERE p.brand = :brand")
    List<Product> findByBrand(@Param("brand") String brand);
    
    @Query("SELECT p FROM Product p WHERE p.title LIKE %:keyword% OR p.description LIKE %:keyword%")
    List<Product> findByKeyword(@Param("keyword") String keyword);
    
    boolean existsByTitle(String title);
}