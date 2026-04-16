package com.vudance.recommendation_system.util;

import io.jsonwebtoken.Jwts;
import io.jsonwebtoken.security.Keys;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import javax.crypto.SecretKey;
import java.util.Date;

@Component
public class JwtUtil {
    
    @Value("${jwt.secret:mySecretKeyForJWTTokenGenerationAndValidationWithMinimum256BitsLength123456}")
    private String jwtSecret;
    
    @Value("${jwt.expiration:86400000}") // 24 hours in milliseconds
    private long jwtExpiration;
    
    /**
     * Generate JWT token for a user
     * @param userId User ID
     * @param email User email
     * @return JWT token
     */
    public String generateToken(Long userId, String email) {
        SecretKey key = Keys.hmacShaKeyFor(jwtSecret.getBytes());
        return Jwts.builder()
                .subject(userId.toString())
                .claim("email", email)
                .issuedAt(new Date())
                .expiration(new Date(System.currentTimeMillis() + jwtExpiration))
                .signWith(key)
                .compact();
    }
    
    /**
     * Extract user ID from JWT token
     * @param token JWT token
     * @return User ID
     */
    public Long extractUserId(String token) {
        SecretKey key = Keys.hmacShaKeyFor(jwtSecret.getBytes());
        return Long.parseLong(Jwts.parser()
                .verifyWith(key)
                .build()
                .parseSignedClaims(token)
                .getPayload()
                .getSubject());
    }
    
    /**
     * Extract email from JWT token
     * @param token JWT token
     * @return Email
     */
    public String extractEmail(String token) {
        SecretKey key = Keys.hmacShaKeyFor(jwtSecret.getBytes());
        return (String) Jwts.parser()
                .verifyWith(key)
                .build()
                .parseSignedClaims(token)
                .getPayload()
                .get("email");
    }
    
    /**
     * Validate JWT token
     * @param token JWT token
     * @return true if token is valid, false otherwise
     */
    public boolean validateToken(String token) {
        try {
            SecretKey key = Keys.hmacShaKeyFor(jwtSecret.getBytes());
            Jwts.parser()
                    .verifyWith(key)
                    .build()
                    .parseSignedClaims(token);
            return true;
        } catch (Exception e) {
            return false;
        }
    }
}
