FROM flink:1.18-java17

WORKDIR /opt/flink

# Jenkins đã build JAR ở folder target/, ta chỉ việc copy vào
# Đảm bảo đường dẫn này đúng với vị trí file JAR sau khi mvn package
COPY target/flink-job-*.jar /opt/flink/usrlib/flink-job.jar

# Các thiết lập khác giữ nguyên
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

USER flink
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["jobmanager"]