# GIAI ĐOẠN 1: Build JAR (Dùng một image có sẵn Maven và JDK)
FROM maven:3.9-eclipse-temurin-17 AS build_stage
WORKDIR /app

USER root
# Copy file cấu hình pom.xml trước để cache dependencies (tối ưu tốc độ build)
COPY pom.xml .
RUN mvn dependency:go-offline -q

# Copy toàn bộ code vào và build
COPY src ./src
RUN mvn clean package -DskipTests -q

# GIAI ĐOẠN 2: Chạy Flink (Chỉ giữ lại file JAR cuối cùng)
FROM flink:1.18-java17
WORKDIR /opt/flink

# Cài đặt curl (nên thực hiện dưới quyền root)
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Copy file JAR
COPY --from=build_stage /app/target/flink-job-1.0-SNAPSHOT.jar /opt/flink/usrlib/flink-job.jar

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["jobmanager"]