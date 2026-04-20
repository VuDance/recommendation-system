# Use Eclipse Temurin JDK 21 as the base image
FROM eclipse-temurin:21-jdk-jammy

# Set the working directory inside the container
WORKDIR /app

# Copy the Maven wrapper and pom.xml
COPY mvnw .
COPY .mvn .mvn
COPY pom.xml .

# Copy the source code
COPY src src

# Build the application
RUN mvn clean package -DskipTests -q

# Create a JAR file
RUN mv target/*.jar app.jar

# Expose the port the app runs on
EXPOSE 8083

# Set JVM options for containerized environments
ENV JAVA_OPTS="-XX:+UseContainerSupport -XX:MaxRAMPercentage=75.0"

# Run the application
ENTRYPOINT ["sh", "-c", "java $JAVA_OPTS -jar app.jar"]