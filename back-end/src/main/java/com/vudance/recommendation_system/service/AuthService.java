package com.vudance.recommendation_system.service;

import com.vudance.recommendation_system.dto.LoginRequest;
import com.vudance.recommendation_system.dto.LoginResponse;
import com.vudance.recommendation_system.dto.RegisterRequest;
import com.vudance.recommendation_system.model.User;
import com.vudance.recommendation_system.repository.UserRepository;
import com.vudance.recommendation_system.util.JwtUtil;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

@Service
public class AuthService {
    
    private final UserRepository userRepository;
    private final JwtUtil jwtUtil;
    private final PasswordEncoder passwordEncoder;
    
    public AuthService(UserRepository userRepository, JwtUtil jwtUtil, PasswordEncoder passwordEncoder) {
        this.userRepository = userRepository;
        this.jwtUtil = jwtUtil;
        this.passwordEncoder = passwordEncoder;
    }
    
    /**
     * Login user and return JWT token
     * @param loginRequest Login request with email and password
     * @return LoginResponse with token
     * @throws RuntimeException if user not found or password is incorrect
     */
    public LoginResponse login(LoginRequest loginRequest) {
        User user = userRepository.findByEmail(loginRequest.getEmail())
                .orElseThrow(() -> new RuntimeException("User not found with email: " + loginRequest.getEmail()));
        
        if (!passwordEncoder.matches(loginRequest.getPassword(), user.getPassword())) {
            throw new RuntimeException("Invalid password");
        }
        
        String token = jwtUtil.generateToken(user.getId(), user.getEmail());
        
        return new LoginResponse(
                user.getId(),
                user.getEmail(),
                user.getFirstName(),
                user.getLastName(),
                token
        );
    }
    
    /**
     * Register a new user
     * @param registerRequest Registration request with user details
     * @return LoginResponse with token
     * @throws RuntimeException if email already exists
     */
    public LoginResponse register(RegisterRequest registerRequest) {
        if (userRepository.findByEmail(registerRequest.getEmail()).isPresent()) {
            throw new RuntimeException("User already exists with email: " + registerRequest.getEmail());
        }
        
        User user = new User();
        user.setEmail(registerRequest.getEmail());
        user.setPassword(passwordEncoder.encode(registerRequest.getPassword()));
        user.setFirstName(registerRequest.getFirstName());
        user.setLastName(registerRequest.getLastName());
        user.setAddress(registerRequest.getAddress());
        
        User savedUser = userRepository.save(user);
        
        String token = jwtUtil.generateToken(savedUser.getId(), savedUser.getEmail());
        
        return new LoginResponse(
                savedUser.getId(),
                savedUser.getEmail(),
                savedUser.getFirstName(),
                savedUser.getLastName(),
                token
        );
    }
}
