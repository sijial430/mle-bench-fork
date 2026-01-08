# RD-Agent Dockerfile for MLE-Bench
# Builds on mlebench-env base image

ARG BASE_IMAGE=mlebench-env:latest
FROM ${BASE_IMAGE}

# Install RD-Agent dependencies
RUN apt-get update && apt-get install -y \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Set up agent directory
ARG AGENT_DIR=/home/agent
WORKDIR ${AGENT_DIR}

# Clone and install RD-Agent from the local copy
# The agent code is copied from the agents/RD-Agent directory
COPY . ${AGENT_DIR}/

# Install RD-Agent package
RUN eval "$(conda shell.bash hook)" && \
    conda activate agent && \
    pip install -e . && \
    pip install python-dotenv typer fire pydantic-settings

# Create necessary directories
ARG SUBMISSION_DIR=/home/submission
ARG LOGS_DIR=/home/logs
ARG CODE_DIR=/home/code

RUN mkdir -p ${SUBMISSION_DIR} ${LOGS_DIR} ${CODE_DIR} && \
    mkdir -p ${AGENT_DIR}/git_ignore_folder/RD-Agent_workspace && \
    mkdir -p ${AGENT_DIR}/logs

# Make start script executable
RUN chmod +x ${AGENT_DIR}/start.sh

# Set environment variables
ENV AGENT_DIR=${AGENT_DIR}
ENV SUBMISSION_DIR=${SUBMISSION_DIR}
ENV LOGS_DIR=${LOGS_DIR}
ENV CODE_DIR=${CODE_DIR}

# Entry point
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["${AGENT_DIR}/start.sh"]
