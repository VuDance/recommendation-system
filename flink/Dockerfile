FROM flink:1.18-java17

WORKDIR /opt/flink

# Chỉ định rõ file JAR để tránh copy nhầm file original
COPY target/flink-job-1.0-SNAPSHOT.jar /opt/flink/usrlib/flink-job.jar

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

USER flink
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["jobmanager"]