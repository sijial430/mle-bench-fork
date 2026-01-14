ARG BASE_IMAGE=mlebench-env
FROM ${BASE_IMAGE}

# Ensure nonroot user exists (required by run.py)
RUN id nonroot 2>/dev/null || useradd -m nonroot

# where to put submission.csv, will be extracted
ARG SUBMISSION_DIR
ENV SUBMISSION_DIR=${SUBMISSION_DIR}
# where to put any logs, will be extracted
ARG LOGS_DIR
ENV LOGS_DIR=${LOGS_DIR}
# where to put any code, will be extracted
ARG CODE_DIR
ENV CODE_DIR=${CODE_DIR}
# where to put any other agent-specific files, will not be necessarily extracted
ARG AGENT_DIR
ENV AGENT_DIR=${AGENT_DIR}

RUN mkdir -p ${LOGS_DIR} ${CODE_DIR} ${AGENT_DIR} ${SUBMISSION_DIR} && \
    chown -R nonroot:nonroot ${LOGS_DIR} ${CODE_DIR} ${AGENT_DIR} ${SUBMISSION_DIR}

ARG CONDA_ENV_NAME=ml-master
ARG REQUIREMENTS=${AGENT_DIR}/requirements.txt

# Copy just the requirements file first for caching
COPY requirements.txt ${AGENT_DIR}/requirements.txt

# Requirements for opencv and other dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create conda environment if it doesn't exist, then install dependencies
RUN if ! conda env list | grep -q "^${CONDA_ENV_NAME} "; then \
        conda create -n ${CONDA_ENV_NAME} python=3.10 -y; \
    fi && \
    conda run -n ${CONDA_ENV_NAME} pip install -r ${AGENT_DIR}/requirements.txt && \
    conda clean -afy

# Put all the agent files in the expected location
COPY . ${AGENT_DIR}

# Make start.sh executable and ensure nonroot owns agent files
RUN chmod +x ${AGENT_DIR}/start.sh && \
    chown -R nonroot:nonroot ${AGENT_DIR}

WORKDIR ${AGENT_DIR}
