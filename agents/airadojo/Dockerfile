FROM mlebench-env

# Install aira-dojo dependencies
WORKDIR /tmp/aira-dojo-setup

# Copy necessary files
COPY aira-dojo-fork/requirements.txt ./requirements.txt
COPY aira-dojo-fork/pyproject.toml ./pyproject.toml
COPY aira-dojo-fork/src ./src

# Install dependencies in the agent conda environment
RUN conda run -n agent pip install -r requirements.txt

# Install aira-dojo package
RUN conda run -n agent pip install -e .

# Clean up
WORKDIR /
RUN rm -rf /tmp/aira-dojo-setup
