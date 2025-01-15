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

def creat_row_data_fram():
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
        simple_logger(f"reading {file_path}")
        with open(file_path, "r") as file:
            content = [float(line.split()[1]) for line in file if line.strip()]

        # Extract the subdirectory names (skip the first three parts: ".", "results", "data")
        subdirs = parts[3:-1]  # Exclude the file name (last part)
        file_name = parts[-1].replace(".txt", "")

        # Combine subdirs, file name, and content into a row
        row = subdirs + [file_name] + content
        data.append(row)
    with open(file_path, "r") as file:
        origin = [str(line.split()[0]) for line in file if line.strip()]

    simple_logger("read all files finish.")
    # Create column names: Subdir1, Subdir2, ..., Name, Content1, Content2, ...
    simple_logger("creating columns ...")
    columns = subdirs + ["Name"] + origin
    simple_logger("create columns done.")

    # Ensure all rows have the same number of columns by padding with None
    padded_data = [row + [None] * (len(columns) - len(row)) for row in data]

    # Create the Polars DataFrame with explicit orientation
    simple_logger("creating row data frame ...")
    df = pl.DataFrame(padded_data, schema=columns, orient="row")
    simple_logger("row data frame created.")
    # Print the DataFrame shape and first few rows
    simple_logger(f"DataFrame shape: {df.shape}")
    return df

def cal_relative(df):
    # Identify the position of the "Name" column
    name_index = df.columns.index("Name")

    # All columns before "Name" are your group-by columns
    groupby_cols = df.columns[:name_index]

    # All columns after "Name" are your numeric columns to calculate relative values
    content_cols = df.columns[name_index + 1:]

    # Group by the subdirectory columns
    grouped_df = df.group_by(groupby_cols)

    # Initialize an empty list to store the results
    relative_data = []

    # Define regex patterns as strings (case-insensitive, with optional numbers)
    ref_pattern = r"(?i)ref\d*"  # (?i) makes it case-insensitive
    dark_pattern = r"(?i)dark\d*"

    # Iterate over each group
    for group in grouped_df:
        group_df = group[1]  # Get the DataFrame for the current group

        # Extract the "ref" and "dark" rows using regex matching
        ref_row = group_df.filter(pl.col("Name").str.contains(ref_pattern))
        dark_row = group_df.filter(pl.col("Name").str.contains(dark_pattern))

        # If "ref" or "dark" rows are missing, skip this group
        if ref_row.is_empty() or dark_row.is_empty():
            continue

        # Extract the values from the "ref" and "dark" rows
        ref_values = ref_row.select(content_cols).to_numpy()[0]
        dark_values = dark_row.select(content_cols).to_numpy()[0]

        # Calculate the relative values for each row in the group
        for row in group_df.iter_rows(named=True):
            # Skip rows that match the "ref" or "dark" patterns
            if not (re.match(ref_pattern, row["Name"], re.IGNORECASE) or re.match(dark_pattern, row["Name"], re.IGNORECASE)):
                other_values = np.array([row[col] for col in content_cols])
                relative_values = (other_values - dark_values) / (ref_values - dark_values)

                # Combine the group-by columns, name, and relative values into a new row
                new_row = [row[col] for col in groupby_cols] + [row["Name"]] + list(relative_values)
                relative_data.append(new_row)

    # Create column names: Subdir1, Subdir2, ..., Name, Content1, Content2, ...
    relative_columns = groupby_cols + ["Name"] + content_cols

    # Create the Polars DataFrame with explicit orientation
    relative_df = pl.DataFrame(relative_data, schema=relative_columns, orient="row")

    return relative_df






def group_and_calculate_mean(df):
    # Group by subdirectories and calculate the mean of Content columns
    # Identify the position of the "Name" column
    name_index = df.columns.index("Name")

    # All columns before "Name" are your group-by columns
    groupby_cols = df.columns[:name_index]

    # All columns after "Name" are your numeric columns to average
    content_cols = df.columns[name_index + 1 :]

    
    grouped_df = (
        df
        .group_by(groupby_cols)  
        .agg([pl.col(col).mean().alias(col) for col in content_cols])
    )

    return grouped_df


if __name__ == "__main__":
#     extractor()
    row_df = creat_row_data_fram()
    # grouped_df = group_and_calculate_mean(row_df)
    relative_df = cal_relative(row_df)
    print(relative_df)