# Recommendation System Backend

A Spring Boot RESTful API for managing users and products with PostgreSQL database integration and Swagger documentation.

## Features

- **RESTful API**: Complete CRUD operations for Users and Products
- **Database Integration**: PostgreSQL with JPA/Hibernate
- **Swagger Documentation**: Auto-generated API documentation
- **Docker Support**: Easy deployment with Docker Compose
- **Validation**: Input validation using Jakarta Bean Validation
- **Pagination**: Support for paginated responses
- **Search**: Full-text search capabilities

## Project Structure

```
src/main/java/com/vudance/recommendation_system/
├── controller/          # REST controllers
│   ├── UserController.java
│   └── ProductController.java
├── service/            # Business logic
│   ├── UserService.java
│   └── ProductService.java
├── repository/         # Data access layer
│   ├── UserRepository.java
│   └── ProductRepository.java
├── model/              # Entity classes
│   ├── User.java
│   └── Product.java
├── dto/                # Data Transfer Objects
│   ├── UserDTO.java
│   └── ProductDTO.java
└── util/               # Utility classes
    └── ModelMapper.java
```

## Prerequisites

- Java 21
- Maven 3.6+
- Docker & Docker Compose (for containerized deployment)
- PostgreSQL (optional, if not using Docker)

## Installation & Setup

### Option 1: Using Docker (Recommended)

1. **Start PostgreSQL Database:**
   ```bash
   docker-compose up -d postgres
   ```

2. **Start the Application:**
   ```bash
   docker-compose --profile app up -d
   ```

3. **Start pgAdmin (Optional):**
   ```bash
   docker-compose --profile dev up -d
   ```

4. **View Logs:**
   ```bash
   docker-compose logs -f app
   ```

5. **Stop Services:**
   ```bash
   docker-compose down
   ```

### Option 2: Local Development

1. **Install Dependencies:**
   ```bash
   mvn clean install
   ```

2. **Start PostgreSQL:**
   - Install PostgreSQL locally or use Docker:
   ```bash
   docker run -d --name postgres-db -e POSTGRES_DB=recommendation_db -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=password123 -p 5432:5432 postgres:16-alpine
   ```

3. **Run the Application:**
   ```bash
   mvn spring-boot:run
   ```

## API Endpoints

### User Management (`/api/users`)

- `GET /api/users` - Get all users
- `GET /api/users/{id}` - Get user by ID
- `GET /api/users/email/{email}` - Get user by email
- `POST /api/users` - Create new user
- `PUT /api/users/{id}` - Update user
- `DELETE /api/users/{id}` - Delete user
- `GET /api/users/search?keyword={keyword}` - Search users

### Product Management (`/api/products`)

- `GET /api/products` - Get all products (with pagination)
- `GET /api/products/{id}` - Get product by ID
- `POST /api/products` - Create new product
- `PUT /api/products/{id}` - Update product
- `DELETE /api/products/{id}` - Delete product
- `GET /api/products/user/{userId}` - Get products by user
- `GET /api/products/category/{category}` - Get products by category
- `GET /api/products/search?keyword={keyword}` - Search products
- `GET /api/products/price-range?minPrice={min}&maxPrice={max}` - Get products by price range

## Swagger Documentation

Access the interactive API documentation at:
- **Swagger UI**: http://localhost:8080/swagger-ui.html
- **OpenAPI JSON**: http://localhost:8080/api-docs

## Database Configuration

The application uses PostgreSQL with the following default settings:

- **Database**: `recommendation_db`
- **Username**: `postgres`
- **Password**: `password123`
- **Port**: `5432`

## Environment Variables

You can override the default configuration using environment variables:

```bash
export SPRING_DATASOURCE_URL=jdbc:postgresql://localhost:5432/your_db
export SPRING_DATASOURCE_USERNAME=your_username
export SPRING_DATASOURCE_PASSWORD=your_password
```

## Testing

Run the test suite:
```bash
mvn test
```

## Docker Commands

### Build the Application
```bash
docker-compose build
```

### View Container Logs
```bash
docker-compose logs -f app
docker-compose logs -f postgres
```

### Access PostgreSQL
```bash
docker exec -it recommendation-postgres psql -U postgres -d recommendation_db
```

### Access pgAdmin
- URL: http://localhost:8081
- Email: admin@example.com
- Password: admin123

## Troubleshooting

### Common Issues

1. **Port 5432 already in use:**
   ```bash
   docker ps | grep postgres
   docker stop <container_id>
   ```

2. **Database connection issues:**
   - Check if PostgreSQL is running
   - Verify database credentials in `application.properties`

3. **Maven build failures:**
   - Ensure Java 21 is installed
   - Check Maven version compatibility

### Health Checks

- Application Health: http://localhost:8080/actuator/health
- Database Health: http://localhost:8080/actuator/health/db

## Development

### Adding New Endpoints

1. Create entity in `model/`
2. Create repository in `repository/`
3. Create service in `service/`
4. Create DTO in `dto/`
5. Create controller in `controller/`
6. Update `ModelMapper` for entity/DTO conversion

### Code Style

The project uses Lombok for reducing boilerplate code. Ensure your IDE has Lombok plugin installed.

## License

This project is licensed under the MIT License.