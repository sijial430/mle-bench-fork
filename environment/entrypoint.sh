#!/bin/bash

# Print commands and their arguments as they are executed
set -x

# log into /home/logs
LOGS_DIR=/home/logs
mkdir -p $LOGS_DIR

{
  # chmod the /home directory such that nonroot users can work on everything within it. We do this at container start
  # time so that anything added later in agent-specific Dockerfiles will also receive the correct permissions.
  # (this command does `chmod a+rw /home` but with the exception of /home/data, which is a read-only volume)
  # Run in background to not block grading server startup
  (find /home -path /home/data -prune -o -exec chmod a+rw {} \; 2>/dev/null &)

  # Quick chmod on essential directories that agent needs immediately
  chmod -R a+rw /home/logs /home/submission 2>/dev/null || true

  ls -l /home

  # Launch grading server, stays alive throughout container lifetime to service agent requests.
  /opt/conda/bin/conda run -n mleb python /private/grading_server.py
} 2>&1 | tee $LOGS_DIR/entrypoint.log
