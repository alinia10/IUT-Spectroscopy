# Use an official Ubuntu base image
FROM ubuntu:22.04

# Set environment variables to avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl git \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Add the deadsnakes PPA for Python 3.12
RUN add-apt-repository ppa:deadsnakes/ppa

# Install Python 3.12 and related tools
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

# Ensure pip is installed for Python 3.12
RUN python3.12 -m ensurepip --upgrade

# Set Python 3.12 as the default Python version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3.12 -

# Add Poetry to PATH
ENV PATH="/root/.local/bin:${PATH}"

# Verify Poetry installation
RUN poetry --version

# Copy project files
COPY pyproject.toml poetry.lock ./

# Configure Poetry to use Python 3.12
RUN poetry env use python3.12

# Install Python dependencies using Poetry
RUN poetry install --no-root

# Copy the rest of the project
COPY . .

# Set up the development environment
CMD ["poetry", "shell"]