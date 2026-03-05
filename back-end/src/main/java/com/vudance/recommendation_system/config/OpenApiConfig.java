package com.vudance.recommendation_system.config;

import io.swagger.v3.oas.models.OpenAPI;
import io.swagger.v3.oas.models.info.Info;
import io.swagger.v3.oas.models.info.Contact;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * OpenAPI/Swagger configuration for the Recommendation System API
 */
@Configuration
public class OpenApiConfig {

    @Bean
    public OpenAPI customOpenAPI() {
        return new OpenAPI()
                .info(new Info()
                        .title("Recommendation System API")
                        .description("RESTful API for managing users and products with Swagger documentation")
                        .version("1.0.0")
                        .contact(new Contact()
                                .name("API Support")
                                .email("support@example.com")
                                .url("https://github.com/VuDance/recommendation-system")));
    }
}