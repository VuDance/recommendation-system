# Multi-stage Dockerfile for Flink Streaming Job

# Stage 1: Builder
FROM maven:3.9-eclipse-temurin-17 as builder

WORKDIR /build

# Copy pom.xml and source code
COPY pom.xml .
COPY dependency-reduced-pom.xml .
COPY src/ ./src/

# Build the project with Maven
RUN mvn clean package -DskipTests -q

# Stage 2: Runtime
FROM flink:1.18-java17

LABEL maintainer="Vudance Team"
LABEL description="Flink streaming job for user view event aggregation"

# Install additional tools if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create directory for job JARs
RUN mkdir -p /opt/flink/usrlib

# Copy the built JAR from builder stage
COPY --from=builder /build/target/flink-job-*.jar /opt/flink/usrlib/

# Set environment variables
ENV FLINK_HOME=/opt/flink

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8081/v1/overview || exit 1

# Set the entrypoint to run the Flink job
ENTRYPOINT ["/docker-entrypoint.sh"]

# Default command to run the job (can be overridden)
CMD ["jobmanager"]