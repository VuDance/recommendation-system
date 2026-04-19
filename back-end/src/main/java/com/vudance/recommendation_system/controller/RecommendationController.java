package com.vudance.recommendation_system.controller;

import com.vudance.recommendation_system.dto.ProductDTO;
import com.vudance.recommendation_system.model.Product;
import com.vudance.recommendation_system.service.ProductService;
import com.vudance.recommendation_system.service.RedisService;
import com.vudance.recommendation_system.util.ModelMapper;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.*;

@RestController
@RequestMapping("/api/recommendations")
@Tag(name = "Recommendations", description = "API for getting product recommendations")
public class RecommendationController {
    private final RedisService redisService;
    private final ProductService productService;
    private final ModelMapper modelMapper;

    public RecommendationController(RedisService redisService, ProductService productService, ModelMapper modelMapper) {
        this.redisService = redisService;
        this.productService = productService;
        this.modelMapper = modelMapper;
    }

    @Operation(summary = "Get user recommendations", description = "Get personalized product recommendations for a user")
    @ApiResponses(value = {
            @ApiResponse(responseCode = "200", description = "Successfully retrieved recommendations"),
            @ApiResponse(responseCode = "404", description = "User not found")
    })
    @GetMapping("/{userId}")
    public ResponseEntity<List<ProductDTO>> getUserRecommendations(@PathVariable String userId) {
        List<String> productIds = (List<String>) redisService.getSortedHashKeys("recommendations:" + userId);
        if (productIds == null || productIds.isEmpty()) {
            List<Product> products = productService.findRandomProducts();
            List<ProductDTO> dto = modelMapper.toProductDTOList(products);
            return ResponseEntity.ok(dto);
        }
        List<Product> products = productService.findAll(productIds);
        List<ProductDTO> dto = modelMapper.toProductDTOList(products);

        return ResponseEntity.ok(dto);
    }

    @Operation(summary = "Get trending products", description = "Get trending products across all users")
    @ApiResponses(value = {
            @ApiResponse(responseCode = "200", description = "Successfully retrieved trending products")
    })
    @GetMapping("/trending")
    public ResponseEntity<List<Map<String, Object>>> getTrendingProducts() {
        List<String> mockAsins = Arrays.asList(
                "0764443682", "1291691480", "1940280001", "1940735033", "1940967805",
                "1942705034", "3293015344", "5378828716", "6041002984", "630456984X"
        );

        List<Map<String, Object>> trending = new ArrayList<>();
        Random random = new Random();

        for (String asin : mockAsins) {
            Map<String, Object> product = new HashMap<>();
            product.put("productId", asin);
            product.put("viewCount", 10 + random.nextInt(90)); // 10-100 views
            product.put("uniqueUsers", 5 + random.nextInt(45)); // 5-50 unique users
            trending.add(product);
        }

        // Sort by view count descending
        trending.sort((a, b) ->
            Integer.compare((Integer) b.get("viewCount"), (Integer) a.get("viewCount"))
        );

        return ResponseEntity.ok(trending);
    }
}