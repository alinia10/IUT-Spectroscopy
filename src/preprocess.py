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


def creat_row_data_fram():
    # Use glob to recursively find all files
    file_paths = glob.glob(os.path.join(extraction_path, "**"), recursive=True)
    data = []

    simple_logger("Starting to read files and create row data frame...")

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
        file_name = re.sub(r"(\d+).txt$", "", parts[-1])  # Remove the last number if it exists

        # Combine subdirs, file name, and content into a row
        row = subdirs + [file_name] + content
        data.append(row)

    # Read the last file to get the origin column names
    with open(file_path, "r") as file:
        origin = [str(line.split()[0]) for line in file if line.strip()]

    simple_logger("Read all files finish.")
    # Create column names: Subdir1, Subdir2, ..., Name, Content1, Content2, ...
    simple_logger("Creating columns ...")
    columns = subdirs + ["Name"] + origin
    simple_logger("Create columns done.")

    # Ensure all rows have the same number of columns by padding with None
    padded_data = [row + [None] * (len(columns) - len(row)) for row in data]

    # Create the Polars DataFrame with explicit orientation
    simple_logger("Creating row data frame ...")
    df = pl.DataFrame(padded_data, schema=columns, orient="row")
    simple_logger("Row data frame created.")
    # Print the DataFrame shape
    simple_logger(f"Row DataFrame shape: {df.shape}")
    return df


def cal_relative(df):
    simple_logger("Starting to calculate relative values...")
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
            simple_logger(f"Skipping group {group[0]} due to missing 'ref' or 'dark' rows.")
            continue

        # Extract the values from the "ref" and "dark" rows
        ref_values = ref_row.select(content_cols).to_numpy()[0]
        dark_values = dark_row.select(content_cols).to_numpy()[0]

        # Calculate the relative values for each row in the group
        for row in group_df.iter_rows(named=True):
            # Skip rows that match the "ref" or "dark" patterns
            if not (re.match(ref_pattern, row["Name"], re.IGNORECASE) or re.match(dark_pattern, row["Name"], re.IGNORECASE)):
                other_values = np.array([row[col] for col in content_cols])
                relative_values = ((other_values - dark_values) * 100 / (ref_values - dark_values))
                relative_values = np.round(relative_values, 2)
                # Combine the group-by columns, name, and relative values into a new row
                new_row = [row[col] for col in groupby_cols] + [row["Name"]] + list(relative_values)
                relative_data.append(new_row)

    # Create column names: Subdir1, Subdir2, ..., Name, Content1, Content2, ...
    relative_columns = groupby_cols + ["Name"] + content_cols

    # Create the Polars DataFrame with explicit orientation
    simple_logger("Creating relative DataFrame...")
    relative_df = pl.DataFrame(relative_data, schema=relative_columns, orient="row")
    sorted_df = relative_df.sort(by=groupby_cols)

    simple_logger("Relative DataFrame created.")
    simple_logger(f"Relative DataFrame shape: {relative_df.shape}")
    return sorted_df

def group_and_calculate_mean(df):
    simple_logger("Starting to group and calculate mean values...")
    
    # Identify the position of the "Name" column
    name_index = df.columns.index("Name")

    # All columns before "Name" are your group-by columns
    groupby_cols = df.columns[:name_index + 1]

    # All columns after "Name" are your numeric columns to average
    content_cols = df.columns[name_index + 1:]

    # Group by the groupby_cols and calculate the mean of replication
    grouped_df = (
        df
        .group_by(groupby_cols)
        .agg(
            [pl.col(col).mean().round(2) for col in content_cols]
        )
    )

    # Sort the grouped DataFrame by the groupby_cols
    sorted_df = grouped_df.sort(by=groupby_cols)

    simple_logger("Grouping, mean calculation, and sorting completed.")
    simple_logger(f"Grouped and sorted DataFrame shape: {sorted_df.shape}")
    return sorted_df

def preprocess(zip_path=None ):
    path_to_zip_file = zip_path if not None else f"./DATA/{zip_name}.zip"
    # Ensure the results directory exists
    os.makedirs("./results", exist_ok=True)

    # Extract the ZIP file
    simple_logger("Starting extraction process...")
    extractor()

    # Create the row DataFrame
    simple_logger("Creating row DataFrame...")
    row_df = creat_row_data_fram()
    # row_df.write_csv("./results/row.csv")
    simple_logger("Row DataFrame saved to ./results/row.csv")

    # Calculate relative values
    simple_logger("Calculating relative values...")
    relative_df = cal_relative(row_df)

    # Group and calculate mean
    simple_logger("Grouping and calculating mean values...")
    grouped_df = group_and_calculate_mean(relative_df)

    # Save the grouped DataFrame to a CSV file
    # row_df.write_csv("./results/imported.csv")
    # relative_df.write_csv("./results/nrelativ.csv")
    grouped_df.write_csv("./results/relative.csv")
    simple_logger("Grouped DataFrame saved to ./results/relativ.csv")

if __name__ == "__main__":
        # Ensure the results directory exists
    os.makedirs("./results", exist_ok=True)

    # Extract the ZIP file
    simple_logger("Starting extraction process...")
    extractor()

    # Create the row DataFrame
    simple_logger("Creating row DataFrame...")
    row_df = creat_row_data_fram()
    # row_df.write_csv("./results/row.csv")
    simple_logger("Row DataFrame saved to ./results/row.csv")

    # Calculate relative values
    simple_logger("Calculating relative values...")
    relative_df = cal_relative(row_df)

    # Group and calculate mean
    simple_logger("Grouping and calculating mean values...")
    grouped_df = group_and_calculate_mean(relative_df)

    # Save the grouped DataFrame to a CSV file
    # row_df.write_csv("./results/imported.csv")
    # relative_df.write_csv("./results/nrelativ.csv")
    grouped_df.write_csv("./results/relativ.csv")
    simple_logger("Grouped DataFrame saved to ./results/relativ.csv")