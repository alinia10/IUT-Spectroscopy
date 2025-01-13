# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory
WORKDIR /workspace

# Install Poetry
RUN pip install poetry

# Copy project files
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry install --no-root

# Copy the rest of the project
COPY . .

# Set up the development environment
CMD ["poetry", "shell"]