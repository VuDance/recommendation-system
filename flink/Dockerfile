# Stage 1: Builder (Chỉ dùng nếu bạn muốn build hoàn toàn trong Docker)
FROM maven:3.9-eclipse-temurin-17 AS builder
WORKDIR /build

# Tận dụng Layer Cache cho Maven Dependencies
COPY pom.xml .
RUN mvn dependency:go-offline -B

COPY src/ ./src/
# Build ra Fat JAR
RUN mvn clean package -DskipTests -q

# Stage 2: Runtime
FROM flink:1.18-java17

# Thông tin metadata
LABEL maintainer="Vudance Team"
LABEL project="recommendation-system"

# Cài đặt công cụ cần thiết & dọn dẹp cache để giảm dung lượng
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# usrlib là folder đặc biệt của Flink, nó sẽ tự động load JAR khi khởi động
USER flink
WORKDIR /opt/flink
RUN mkdir -p /opt/flink/usrlib

# Copy file JAR (Lấy tên cụ thể để tránh copy nhầm file JAR phụ)
COPY --from=builder /build/target/flink-job-1.0-SNAPSHOT.jar /opt/flink/usrlib/flink-job.jar

# Health check - Kiểm tra xem JobManager có sống không
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8081/v1/overview || exit 1

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["jobmanager"]