FROM flink:1.18-java17

# Copy the Flink job JAR
COPY target/flink-job-1.0-SNAPSHOT.jar /opt/flink/usrlib/

# Set the entrypoint to run the Flink job
ENTRYPOINT ["/docker-entrypoint.sh"]

# Default command to run the job (can be overridden)
CMD ["jobmanager"]