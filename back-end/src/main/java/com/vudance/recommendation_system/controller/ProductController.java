package com.vudance.recommendation_system.controller;

import com.vudance.recommendation_system.dto.ProductDTO;
import com.vudance.recommendation_system.model.Product;
import com.vudance.recommendation_system.service.ProductService;
import com.vudance.recommendation_system.service.EventPublisherService;
import com.vudance.recommendation_system.util.ModelMapper;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/products")
@Tag(name = "Product Management", description = "API for managing products")
public class ProductController {
    
    private final ProductService productService;
    private final ModelMapper modelMapper;
    private final EventPublisherService eventPublisherService;
    
    public ProductController(ProductService productService, ModelMapper modelMapper, EventPublisherService eventPublisherService) {
        this.productService = productService;
        this.modelMapper = modelMapper;
        this.eventPublisherService = eventPublisherService;
    }
    
    @Operation(summary = "Get all products", description = "Retrieve a list of all products with pagination")
    @ApiResponses(value = {
        @ApiResponse(responseCode = "200", description = "Successfully retrieved products")
    })
    @GetMapping
    public ResponseEntity<Page<ProductDTO>> getAllProducts(
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "10") int size) {
        
        Pageable pageable = PageRequest.of(page, size);
        Page<Product> products = productService.findAll(pageable);
        Page<ProductDTO> productDTOs = products.map(modelMapper::toProductDTO);
        
        return ResponseEntity.ok(productDTOs);
    }
    
    @Operation(summary = "Get product by ID", description = "Retrieve a product by its ID")
    @ApiResponses(value = {
        @ApiResponse(responseCode = "200", description = "Successfully retrieved product"),
        @ApiResponse(responseCode = "404", description = "Product not found")
    })
    @GetMapping("/{id}")
    public ResponseEntity<ProductDTO> getProductById(@PathVariable String id) {
        // Extract user ID from JWT token if authenticated
        String userId = null;
        Object principal = SecurityContextHolder.getContext().getAuthentication().getPrincipal();
        if (principal instanceof Long) {
            userId = String.valueOf(principal);
        }
        
        // Publish view event to Kafka
        eventPublisherService.publishUserViewEvent(userId, id);
        return productService.findById(id)
                .map(product -> ResponseEntity.ok(modelMapper.toProductDTO(product)))
                .orElse(ResponseEntity.notFound().build());
    }
    @ApiResponses(value = {
        @ApiResponse(responseCode = "200", description = "Successfully retrieved products")
    })
    @GetMapping("/category/{category}")
    public ResponseEntity<List<ProductDTO>> getProductsByCategory(@PathVariable String category) {
        List<Product> products = productService.findByCategory(category);
        List<ProductDTO> productDTOs = products.stream()
                .map(modelMapper::toProductDTO)
                .collect(Collectors.toList());
        
        return ResponseEntity.ok(productDTOs);
    }
    
    @Operation(summary = "Search products", description = "Search products by keyword in name or description")
    @ApiResponses(value = {
        @ApiResponse(responseCode = "200", description = "Successfully retrieved search results")
    })
    @GetMapping("/search")
    public ResponseEntity<List<ProductDTO>> searchProducts(@RequestParam String keyword) {
        List<Product> products = productService.searchProducts(keyword);
        List<ProductDTO> productDTOs = products.stream()
                .map(modelMapper::toProductDTO)
                .collect(Collectors.toList());
        
        return ResponseEntity.ok(productDTOs);
    }
}