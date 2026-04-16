# JWT Authentication Guide

## Overview
This guide explains how to use the JWT (JSON Web Token) authentication system in the Recommendation System application.

## Backend Setup

### 1. JWT Dependencies
The following dependencies have been added to `pom.xml`:
- `jjwt-api` - JWT API
- `jjwt-impl` - JWT implementation
- `jjwt-jackson` - JSON serialization support
- `spring-boot-starter-security` - Spring Security

### 2. Database Tables
A new `users` table has been created with the following fields:
- `id` - Auto-generated primary key
- `email` - User email (unique)
- `password` - Encrypted password (BCrypt)
- `firstName` - First name
- `lastName` - Last name
- `address` - Address (optional)
- `createdAt` - Creation timestamp
- `updatedAt` - Last update timestamp

### 3. JWT Configuration
Configuration is in `application.properties`:
```properties
jwt.secret=mySecretKeyForJWTTokenGenerationAndValidationWithMinimum256BitsLength123456
jwt.expiration=86400000  # 24 hours in milliseconds
```

## API Endpoints

### Login Endpoint
**POST** `/api/auth/login`

Request body:
```json
{
  "email": "user@example.com",
  "password": "password123"
}
```

Response:
```json
{
  "userId": 1,
  "email": "user@example.com",
  "firstName": "John",
  "lastName": "Doe",
  "token": "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIiwiZW1haWwiOiJ..."
}
```

## Backend: Extracting User ID from JWT

### 1. Using JwtUtil Class
```java
@Component
public class YourService {
    
    @Autowired
    private JwtUtil jwtUtil;
    
    public void example(String token) {
        Long userId = jwtUtil.extractUserId(token);
        String email = jwtUtil.extractEmail(token);
        boolean isValid = jwtUtil.validateToken(token);
    }
}
```

### 2. Using SecurityContextHolder
```java
@RestController
@RequestMapping("/api/example")
public class ExampleController {
    
    @GetMapping("/user-info")
    public ResponseEntity<?> getUserInfo() {
        // Get the authenticated user ID from SecurityContext
        Authentication auth = SecurityContextHolder.getContext().getAuthentication();
        Long userId = (Long) auth.getPrincipal();
        
        return ResponseEntity.ok("User ID: " + userId);
    }
}
```

### 3. Using @AuthenticationPrincipal Annotation
```java
@RestController
@RequestMapping("/api/example")
public class ExampleController {
    
    @GetMapping("/user-info")
    public ResponseEntity<?> getUserInfo(@AuthenticationPrincipal Long userId) {
        return ResponseEntity.ok("User ID: " + userId);
    }
}
```

## Frontend: JWT Token Management

### 1. Login User
```typescript
import api from '../services/api'

const loginResponse = await api.post('/auth/login', {
  email: 'user@example.com',
  password: 'password123'
})

// Store token and user info
localStorage.setItem('token', loginResponse.data.token)
localStorage.setItem('userId', loginResponse.data.userId)
localStorage.setItem('email', loginResponse.data.email)

// Set authorization header
api.defaults.headers.common['Authorization'] = `Bearer ${loginResponse.data.token}`
```

### 2. Extract User ID from JWT Token
```typescript
import authService from '../services/auth-service'

// Method 1: Get from localStorage (immediately after login)
const userId = localStorage.getItem('userId')

// Method 2: Extract from stored token
const token = localStorage.getItem('token')
const userId = authService.extractUserIdFromToken(token)

// Method 3: Use auth service helper
const userInfo = authService.getUserInfo()
console.log(userInfo.userId)
```

### 3. Using Auth Service
```typescript
import authService from '../services/auth-service'

// Check if user is authenticated
if (authService.isAuthenticated()) {
  const userId = authService.getUserId()
  const userInfo = authService.getUserInfo()
}

// Logout
authService.logout()
```

## JWT Token Structure

JWT tokens have 3 parts separated by dots:
```
eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIiwiZW1haWwiOiJ1c2VyQGV4YW1wbGUuY29tIn0.TJVA95OrM7E2cBab30RMHrHDcEfxjoYZgeFONFh7HgQ
```

Parts:
1. **Header** - Algorithm and token type (Base64 encoded)
2. **Payload** - Claims including user ID in `sub` field (Base64 encoded)
3. **Signature** - HMAC signature for verification

### Decoding Payload
```typescript
const token = "your_jwt_token_here"
const parts = token.split('.')
const payload = JSON.parse(atob(parts[1]))
console.log(payload.sub)  // User ID
console.log(payload.email)  // Email
```

## Security Headers

When making authenticated requests, always include the Authorization header:
```
Authorization: Bearer eyJhbGciOiJIUzI1NiJ9...
```

The axios instance in `src/services/api.ts` is configured to automatically include this header for all requests after login.

## CORS Configuration

CORS is configured in `SecurityConfig.java` to allow:
- Origins: `http://localhost:5173`, `http://localhost:3000`
- Methods: GET, POST, PUT, DELETE, OPTIONS
- Headers: * (all headers)
- Credentials: Allowed

## Testing

### Using Swagger UI
1. Navigate to `http://localhost:8080/swagger-ui.html`
2. Use the `/api/auth/login` endpoint to login
3. Copy the token from the response
4. Click "Authorize" button and paste: `Bearer <your_token>`
5. Test other endpoints with authorization

### Using cURL
```bash
# Login
curl -X POST http://localhost:8080/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"user@example.com","password":"password123"}'

# Use token in subsequent requests
curl -X GET http://localhost:8080/api/protected-endpoint \
  -H "Authorization: Bearer your_token_here"
```

## Common Issues

### 1. Invalid Signature
- Token was not created with the same secret key
- Check `jwt.secret` configuration

### 2. Token Expired
- Token expiration time has passed
- User needs to login again to get a new token

### 3. Missing Authorization Header
- Requests to protected endpoints must include the Authorization header
- Format: `Authorization: Bearer <token>`

### 4. CORS Error
- Check CORS configuration in `SecurityConfig.java`
- Ensure your frontend origin is in the allowed origins list

## Database Setup

To create a test user, you can:

1. **Via SQL:**
```sql
INSERT INTO users (email, password, first_name, last_name, address, created_at, updated_at)
VALUES ('test@example.com', '$2a$10$...', 'John', 'Doe', 'Address', NOW(), NOW());
```

Note: Password should be BCrypt hashed. You can use an online BCrypt generator.

2. **Via API** (create a register endpoint if needed):
```typescript
POST /api/auth/register
{
  "email": "user@example.com",
  "password": "password123",
  "firstName": "John",
  "lastName": "Doe",
  "address": "123 Main St"
}
```

## Next Steps

1. Add user registration endpoint
2. Implement refresh token mechanism
3. Add role-based access control (RBAC)
4. Implement password reset functionality
5. Add email verification
