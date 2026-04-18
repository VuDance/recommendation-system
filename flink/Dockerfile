FROM flink:1.18-java17

WORKDIR /opt/flink

# Dùng đường dẫn tương đối từ Build Context (thư mục flink/)
COPY target/flink-job-*.jar /opt/flink/usrlib/flink-job.jar

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

USER flink
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["jobmanager"]