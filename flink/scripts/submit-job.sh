#!/bin/bash
set -e

RUNNING=$(curl -s http://jobmanager:8081/v1/jobs | grep -o '"status":"RUNNING"' | wc -l)

if [ "$RUNNING" -eq '0' ]; then
  echo "Submitting job..."
  /opt/flink/bin/flink run -m jobmanager:8081 /opt/flink/usrlib/flink-job.jar
else
  echo "Job already running, skip."
fi