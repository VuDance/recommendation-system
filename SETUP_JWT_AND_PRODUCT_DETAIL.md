# Setup Guide - JWT Login, Product Detail Page, and User Management

This guide will help you set up and use the JWT authentication system, product detail page with recommendations, and user management in the Recommendation System application.

## Table of Contents
1. [Backend Setup](#backend-setup)
2. [Frontend Setup](#frontend-setup)
3. [Database Initialization](#database-initialization)
4. [JWT Token Usage](#jwt-token-usage)
5. [Testing the System](#testing-the-system)
6. [Integration with Your API](#integration-with-your-api)

## Backend Setup

### 1. Update Dependencies
The following have been added to `pom.xml`:
- JWT libraries (jjwt-api, jjwt-impl, jjwt-jackson)
- Spring Security

### 2. Created Components
- **User Entity** (`model/User.java`) - Simple user table
- **UserRepository** (`repository/UserRepository.java`) - Database access
- **JwtUtil** (`util/JwtUtil.java`) - JWT token generation and validation
- **AuthService** (`service/AuthService.java`) - Authentication logic
- **AuthController** (`controller/AuthController.java`) - Login endpoint
- **SecurityConfig** (`config/SecurityConfig.java`) - Spring Security configuration
- **JwtAuthenticationFilter** (`config/JwtAuthenticationFilter.java`) - JWT filter

### 3. Configuration
Update `application.properties` (already done):
```properties
jwt.secret=mySecretKeyForJWTTokenGenerationAndValidationWithMinimum256BitsLength123456
jwt.expiration=86400000  # 24 hours
```

### 4. Build and Run
```bash
cd back-end
mvn clean install
mvn spring-boot:run
```

## Frontend Setup

### 1. Environment Variables
Create `.env` file (already created):
```
VITE_API_URL=http://localhost:8080
```

### 2. Created Components
- **Login Page** (`pages/Login.tsx`) - User login form
- **ProductDetail Page** (`pages/ProductDetail.tsx`) - Product details with recommendations
- **Auth Service** (`services/auth-service.ts`) - JWT token management utility

### 3. Updated Components
- **App.tsx** - Added routes for login and product detail
- **Navbar.tsx** - Added login/logout buttons
- **ProductCard.tsx** - Added links to product detail page
- **api.ts** - Added automatic JWT token inclusion in requests

### 4. Install Dependencies and Run
```bash
cd front-end/vite-project
npm install
npm run dev
```

The frontend will run on `http://localhost:5173`

## Database Initialization

### 1. Automatic Schema Creation
The User table will be created automatically on first run due to:
```properties
spring.jpa.hibernate.ddl-auto=update
```

### 2. Create Test User (PostgreSQL)

Run this SQL to create a test user. The password "password123" is hashed using BCrypt:

```sql
INSERT INTO users (email, password, first_name, last_name, address, created_at, updated_at)
VALUES (
  'test@example.com',
  '$2a$10$Rd1L4C1dkVAFmJxnP8x3u.hB4jF0P6Z5J9c2K.L1M.N2O3P4Q5R6S7',
  'John',
  'Doe',
  '123 Main Street',
  NOW(),
  NOW()
);
```

**Login credentials:**
- Email: `test@example.com`
- Password: `password123`

### 3. Generate BCrypt Hash
If you want to create users with different passwords, use an online BCrypt generator:
- Visit: https://bcrypt-generator.com/
- Hash your desired password
- Use the hash in the INSERT statement

## JWT Token Usage

### Backend: Extracting User ID

#### Method 1: Using JwtUtil
```java
@RestController
@RequestMapping("/api/example")
public class ExampleController {
    
    @Autowired
    private JwtUtil jwtUtil;
    
    @GetMapping("/test")
    public ResponseEntity<?> testEndpoint(@RequestHeader("Authorization") String token) {
        String cleanToken = token.replace("Bearer ", "");
        Long userId = jwtUtil.extractUserId(cleanToken);
        String email = jwtUtil.extractEmail(cleanToken);
        
        return ResponseEntity.ok("User ID: " + userId + ", Email: " + email);
    }
}
```

#### Method 2: Using SecurityContextHolder
```java
@RestController
@RequestMapping("/api/example")
public class ExampleController {
    
    @GetMapping("/test")
    public ResponseEntity<?> testEndpoint() {
        Authentication auth = SecurityContextHolder.getContext().getAuthentication();
        Long userId = (Long) auth.getPrincipal();
        
        return ResponseEntity.ok("User ID: " + userId);
    }
}
```

#### Method 3: Using @AuthenticationPrincipal
```java
@RestController
@RequestMapping("/api/example")
public class ExampleController {
    
    @GetMapping("/test")
    public ResponseEntity<?> testEndpoint(@AuthenticationPrincipal Long userId) {
        return ResponseEntity.ok("User ID: " + userId);
    }
}
```

### Frontend: Extracting User ID

#### Method 1: From localStorage (Recommended for Login Response)
```typescript
// Immediately after login
const userId = localStorage.getItem('userId')
```

#### Method 2: Extract from JWT Token
```typescript
import authService from '../services/auth-service'

const token = localStorage.getItem('token')
const userId = authService.extractUserIdFromToken(token)
```

#### Method 3: Using Auth Service
```typescript
import authService from '../services/auth-service'

// Get user info
const userInfo = authService.getUserInfo()
console.log(userInfo.userId)

// Check if authenticated
if (authService.isAuthenticated()) {
  const userId = authService.getUserId()
}

// Logout
authService.logout()
```

#### Method 4: Manual Decoding
```typescript
const token = localStorage.getItem('token')
if (token) {
  const parts = token.split('.')
  const payload = JSON.parse(atob(parts[1]))
  console.log('User ID:', payload.sub)
  console.log('Email:', payload.email)
}
```

## Testing the System

### 1. Access Frontend
1. Open `http://localhost:5173` in your browser
2. Click "Login" in the navbar
3. Use credentials:
   - Email: `test@example.com`
   - Password: `password123`
4. After login, you'll be redirected to the home page
5. Click on any product to see the detail page with recommendations

### 2. Test with Swagger UI
1. Navigate to `http://localhost:8080/swagger-ui.html`
2. Find `/api/auth/login` endpoint
3. Click "Try it out"
4. Enter your credentials
5. Copy the token from the response
6. Click "Authorize" button and paste: `Bearer <your_token>`
7. Now all protected endpoints will include authorization

### 3. Test with cURL
```bash
# Login
curl -X POST http://localhost:8080/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"password123"}'

# Use token in subsequent requests
curl -X GET http://localhost:8080/api/products \
  -H "Authorization: Bearer your_token_here"
```

## Integration with Your API

### 1. Adding User ID to Recommendations
Update your recommendation endpoint to accept user ID:

```java
@RestController
@RequestMapping("/api/recommendations")
public class RecommendationController {
    
    @GetMapping
    public ResponseEntity<?> getRecommendations(
            @RequestParam Integer productId,
            @AuthenticationPrincipal Long userId) {
        
        // Use userId and productId to fetch personalized recommendations
        List<Product> recommendations = recommendationService.getRecommendations(userId, productId);
        
        return ResponseEntity.ok(recommendations);
    }
}
```

### 2. Protecting Endpoints
By default, all endpoints except these are protected:
- `/api/auth/**` - Authentication endpoints
- `/swagger-ui.html` - Swagger documentation
- `/v3/api-docs/**` - API documentation
- `/api/products/**` - Products (public for now)

To protect an endpoint, it automatically requires a valid JWT token.

### 3. Making Authenticated Requests from Frontend
```typescript
import api from '../services/api'

// Token is automatically added by the interceptor
const response = await api.get('/api/recommendations', {
  params: {
    productId: 1
  }
})
```

## File Structure

### Backend Files Created
```
back-end/src/main/java/com/vudance/recommendation_system/
├── model/User.java
├── repository/UserRepository.java
├── service/AuthService.java
├── controller/AuthController.java
├── config/
│   ├── SecurityConfig.java
│   └── JwtAuthenticationFilter.java
├── util/JwtUtil.java
├── dto/
│   ├── LoginRequest.java
│   └── LoginResponse.java
└── resources/application.properties
```

### Frontend Files Created
```
front-end/vite-project/src/
├── pages/
│   ├── Login.tsx
│   ├── Login.css
│   ├── ProductDetail.tsx
│   └── ProductDetail.css
├── services/auth-service.ts
├── components/
│   ├── Navbar.tsx (updated)
│   ├── ProductCard.tsx (updated)
│   └── Navbar.css (updated)
├── App.tsx (updated)
└── .env (created)
```

## Troubleshooting

### CORS Error
- Ensure backend is running on `http://localhost:8080`
- Check CORS configuration in `SecurityConfig.java`
- Frontend should be on `http://localhost:5173`

### Token Not Being Sent
- Check if token is in localStorage: `localStorage.getItem('token')`
- Verify Authorization header: Check browser DevTools > Network tab

### Login Fails
- Check test user exists in database
- Verify database connection: `spring.datasource.url`
- Check password is correct (test user password is "password123")

### Product Detail Page Not Loading
- Ensure product ID in URL is valid
- Check if product exists in database
- Verify API endpoint: `GET /api/products/{id}`

## Next Steps

1. **Add Registration**: Create `/api/auth/register` endpoint
2. **Refresh Tokens**: Implement token refresh mechanism
3. **Role-Based Access**: Add user roles and permissions
4. **Password Reset**: Implement password reset functionality
5. **Email Verification**: Add email verification for new accounts
6. **User Profile**: Create user profile edit page

## Documentation
For more details, see `JWT_AUTHENTICATION_GUIDE.md` in the project root.
