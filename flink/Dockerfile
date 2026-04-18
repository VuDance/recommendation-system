# GIAI ĐOẠN 1: Build JAR (Dùng một image có sẵn Maven và JDK)
FROM maven:3.9-eclipse-temurin-17 AS build_stage
WORKDIR /app

# Copy file cấu hình pom.xml trước để cache dependencies (tối ưu tốc độ build)
COPY pom.xml .
RUN mvn dependency:go-offline -q

# Copy toàn bộ code vào và build
COPY src ./src
RUN mvn clean package -DskipTests -q

# GIAI ĐOẠN 2: Chạy Flink (Chỉ giữ lại file JAR cuối cùng)
FROM flink:1.18-java17
WORKDIR /opt/flink

# Copy file JAR từ build_stage sang. 
# Lưu ý: Kiểm tra chính xác tên file JAR trong thư mục target của container build
COPY --from=build_stage /app/target/flink-job-1.0-SNAPSHOT.jar /opt/flink/usrlib/flink-job.jar

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

USER flink
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["jobmanager"]