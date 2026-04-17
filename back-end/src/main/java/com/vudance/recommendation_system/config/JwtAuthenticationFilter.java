package com.vudance.recommendation_system.config;

import com.vudance.recommendation_system.util.JwtUtil;
import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.web.filter.OncePerRequestFilter;

import java.io.IOException;
import java.util.List;

public class JwtAuthenticationFilter extends OncePerRequestFilter {
    
    private final JwtUtil jwtUtil;
    
    public JwtAuthenticationFilter(JwtUtil jwtUtil) {
        this.jwtUtil = jwtUtil;
    }
    
    @Override
protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response, FilterChain filterChain)
        throws ServletException, IOException {
    
    String token = getTokenFromRequest(request);
    System.out.println("🔍 Token: " + token);
    System.out.println("🔍 Valid: " + (token != null && jwtUtil.validateToken(token)));
    
    if (token != null && jwtUtil.validateToken(token)) {
        Long userId = jwtUtil.extractUserId(token);
        System.out.println("🔍 UserId: " + userId);
        UsernamePasswordAuthenticationToken authentication = 
                new UsernamePasswordAuthenticationToken(userId, null, List.of());
        SecurityContextHolder.getContext().setAuthentication(authentication);
        System.out.println("🔍 Authentication set: " + SecurityContextHolder.getContext().getAuthentication());
    }
    
    filterChain.doFilter(request, response);
}
    
    private String getTokenFromRequest(HttpServletRequest request) {
        String bearerToken = request.getHeader("Authorization");
        if (bearerToken != null && bearerToken.startsWith("Bearer ")) {
            return bearerToken.substring(7);
        }
        return null;
    }
}
