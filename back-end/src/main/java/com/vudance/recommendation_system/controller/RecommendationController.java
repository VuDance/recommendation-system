package com.vudance.recommendation_system.controller;

import com.vudance.recommendation_system.dto.RecommendationDTO;
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

    @Operation(summary = "Get user recommendations", description = "Get personalized product recommendations for a user")
    @ApiResponses(value = {
            @ApiResponse(responseCode = "200", description = "Successfully retrieved recommendations"),
            @ApiResponse(responseCode = "404", description = "User not found")
    })
    @GetMapping("/{userId}")
    public ResponseEntity<List<RecommendationDTO>> getUserRecommendations(@PathVariable String userId) {
        // Mock recommendation data based on real ASINs from products_clean.csv
        List<String> mockAsins = Arrays.asList(
                "0764443682",  // Slime Time Fall Fest
                "1291691480",  // XCC Qi promise new spider snake
                "1940280001",  // Magical Things I Really Do Do Too!
                "1940735033",  // Ashes to Ashes, Oranges to Oranges
                "1940967805",  // Aether & Empire #1
                "1942705034",  // 365 Affirmations for a Year of Love
                "3293015344",  // Blessed by Pope Benedetto XVI Bracelet
                "5378828716",  // Womens Sexy Sleeveless Camouflage Print
                "6041002984",  // Sevendayz Men's Shady Records Eminem Hoodie
                "630456984X",  // Dante's Peak - Laserdisc
                "7106116521",  // Milliongadgets Earring Safety Backs
                "8037200124",  // Envirosax Kids Series Jessie & Lulu
                "8037200221",  // Envirosax Greengrocer Series Bag
                "8279996567",  // Blessed by Pope Benedetto XVI Rosary
                "9239282785"   // Tideclothes ALAGIRLS Strapless Beading
        );

        // Shuffle and select random recommendations for the user
        Collections.shuffle(mockAsins, new Random(userId.hashCode()));
        int numRecommendations = Math.min(5 + (userId.hashCode() % 5), mockAsins.size()); // 5-9 recommendations

        List<RecommendationDTO> recommendations = new ArrayList<>();
        long windowStart = System.currentTimeMillis() - (60 * 60 * 1000); // 1 hour ago
        long windowEnd = System.currentTimeMillis();

        for (int i = 0; i < numRecommendations; i++) {
            RecommendationDTO rec = new RecommendationDTO();
            rec.setUserId(userId);
            Map<String, Integer> viewedProducts = new HashMap<>();
            viewedProducts.put(mockAsins.get(i), 1 + (i % 3)); // 1-3 views
            rec.setViewedProducts(viewedProducts);
            rec.setWindowStart(windowStart);
            rec.setWindowEnd(windowEnd);
            recommendations.add(rec);
        }

        return ResponseEntity.ok(recommendations);
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