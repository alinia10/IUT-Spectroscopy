#!/usr/bin/env python3

import re
import polars as pl
import glob
import numpy as np
from zipfile import ZipFile
import os

from conf import config
from log import simple_logger

zip_name = config["inputs"]["zip_name"]
path_to_zip_file = f"./DATA/{zip_name}.zip"
extraction_path = f"./results/data"


def extractor():
    with ZipFile(path_to_zip_file, 'r') as zip_ref:
        simple_logger(f"start extracting {zip_name} to results folder ... ")
        zip_ref.extractall(extraction_path)
        simple_logger(f"extract completed.")


def path_to_list():
    ...


def get_dark_ref():
    ...


def mean_of_units():
    ...


# Use glob to recursively find all files
file_paths = glob.glob(os.path.join(extraction_path, "**"), recursive=True)
data = []

# Process each file
for file_path in file_paths:
    # Skip directories
    if os.path.isdir(file_path):
        continue

    # Split the file path into its components
    parts = os.path.normpath(file_path).split(os.sep)

    # Read the content of the file (second column)
    with open(file_path, "r") as file:
        content = [float(line.split()[1]) for line in file if line.strip()]

    # Extract the subdirectory names (skip the first three parts: ".", "results", "data")
    subdirs = parts[3:-1]  # Exclude the file name (last part)
    file_name = parts[-1].replace(".txt", "")

    # Combine subdirs, file name, and content into a row
    row = subdirs + [file_name] + content
    data.append(row)

# Determine the maximum number of subdirectories to define column names
max_subdirs = max(len(row) - len(content) - 1 for row in data)  # Subtract for "Name" and content

# Determine the maximum number of content columns
max_content_cols = max(len(row) - max_subdirs - 1 for row in data)

# Create column names: Subdir1, Subdir2, ..., Name, Content1, Content2, ...
columns = [f"Subdir{i+1}" for i in range(max_subdirs)] + ["Name"] + [f"Content{i+1}" for i in range(max_content_cols)]

# Ensure all rows have the same number of columns by padding with None
padded_data = [row + [None] * (len(columns) - len(row)) for row in data]

# Create the Polars DataFrame with explicit orientation
df = pl.DataFrame(padded_data, schema=columns, orient="row")

# Print the DataFrame shape and first few rows
print(f"DataFrame shape: {df.shape}")
print(df.head())

# if __name__ == "__main__":
#     extractor()