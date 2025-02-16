#!/usr/bin/env python3
import os
import re
import glob
import numpy as np
from zipfile import ZipFile
import polars as pl

from conf import config
from log import simple_logger

# Expected ZIP file name from configuration
zip_name = config["inputs"]["zip_name"]
expected_zip_filename = f"{zip_name}.zip"
expected_zip_path = f"./DATA/{expected_zip_filename}"
extraction_path = f"./results/data"

def ensure_zip_file(provided_zip_path=None):
    """
    Ensure a ZIP file with the expected name exists in the DATA folder.
    If provided_zip_path is given and its basename differs from expected,
    rename it. Otherwise, search for any ZIP file in ./DATA.
    """
    if provided_zip_path:
        if os.path.basename(provided_zip_path) != expected_zip_filename:
            simple_logger(f"Renaming {provided_zip_path} to {expected_zip_path}")
            os.rename(provided_zip_path, expected_zip_path)
        return expected_zip_path
    else:
        if os.path.exists(expected_zip_path):
            return expected_zip_path
        else:
            files = [f for f in os.listdir('./DATA') if f.endswith('.zip')]
            if files:
                found_zip = os.path.join('./DATA', files[0])
                simple_logger(f"Renaming {found_zip} to {expected_zip_path}")
                os.rename(found_zip, expected_zip_path)
                return expected_zip_path
            else:
                raise FileNotFoundError("No ZIP file found in ./DATA folder.")

def extractor(zip_file_path):
    with ZipFile(zip_file_path, 'r') as zip_ref:
        simple_logger(f"Start extracting {expected_zip_filename} to {extraction_path} ...")
        zip_ref.extractall(extraction_path)
        simple_logger("Extraction completed.")

def creat_row_data_fram():
    file_paths = glob.glob(os.path.join(extraction_path, "**"), recursive=True)
    data = []
    simple_logger("Starting to read files and create row data frame...")
    for file_path in file_paths:
        if os.path.isdir(file_path):
            continue
        parts = os.path.normpath(file_path).split(os.sep)
        simple_logger(f"Reading {file_path}")
        with open(file_path, "r") as file:
            content = [float(line.split()[1]) for line in file if line.strip()]
        subdirs = parts[3:-1]
        file_name = re.sub(r"(\d+).txt$", "", parts[-1])
        row = subdirs + [file_name] + content
        data.append(row)
    # Use the last file to retrieve the original wavelength labels
    with open(file_path, "r") as file:
        origin = [str(line.split()[0]) for line in file if line.strip()]
    simple_logger("Finished reading files.")
    columns = subdirs + ["Name"] + origin
    simple_logger("Creating DataFrame columns...")
    df = pl.DataFrame(data, schema=columns, orient="row")
    simple_logger(f"Row DataFrame shape: {df.shape}")
    return df

def cal_relative(df):
    simple_logger("Calculating relative values...")
    name_index = df.columns.index("Name")
    groupby_cols = df.columns[:name_index]
    content_cols = df.columns[name_index + 1:]
    grouped_df = df.group_by(groupby_cols)
    relative_data = []
    ref_pattern = r"(?i)ref\d*"
    dark_pattern = r"(?i)dark\d*"
    for group in grouped_df:
        group_df = group[1]
        ref_row = group_df.filter(pl.col("Name").str.contains(ref_pattern))
        dark_row = group_df.filter(pl.col("Name").str.contains(dark_pattern))
        if ref_row.is_empty() or dark_row.is_empty():
            simple_logger(f"Skipping group {group[0]} due to missing 'ref' or 'dark' rows.")
            continue
        ref_values = ref_row.select(content_cols).to_numpy()[0]
        dark_values = dark_row.select(content_cols).to_numpy()[0]
        for row in group_df.iter_rows(named=True):
            if re.match(ref_pattern, row["Name"], re.IGNORECASE) or re.match(dark_pattern, row["Name"], re.IGNORECASE):
                continue
            other_values = np.array([row[col] for col in content_cols])
            relative_values = ((other_values - dark_values) * 100 / (ref_values - dark_values))
            relative_values = np.round(relative_values, 2)
            new_row = [row[col] for col in groupby_cols] + [row["Name"]] + list(relative_values)
            relative_data.append(new_row)
    relative_columns = groupby_cols + ["Name"] + content_cols
    simple_logger("Creating relative DataFrame...")
    relative_df = pl.DataFrame(relative_data, schema=relative_columns, orient="row")
    sorted_df = relative_df.sort(by=groupby_cols)
    simple_logger(f"Relative DataFrame shape: {relative_df.shape}")
    return sorted_df

def group_and_calculate_mean(df):
    simple_logger("Grouping and calculating mean values...")
    name_index = df.columns.index("Name")
    groupby_cols = df.columns[:name_index+1]
    content_cols = df.columns[name_index+1:]
    grouped_df = df.group_by(groupby_cols).agg([pl.col(col).mean().round(2) for col in content_cols])
    sorted_df = grouped_df.sort(by=groupby_cols)
    simple_logger(f"Grouped DataFrame shape: {sorted_df.shape}")
    return sorted_df

# --- New helper functions for wavelength aggregation ---

def is_within_target(col_name, target, tol=0.5):
    try:
        value = float(col_name)
        return (target - tol) <= value <= (target + tol)
    except ValueError:
        return False

def get_wavelength_mean_expr(df, target, tol=0.5):
    """
    Returns a Polars expression representing the row-wise mean of all columns
    whose names (converted to float) lie within target ± tol.
    """
    selected_cols = [c for c in df.columns if is_within_target(c, target, tol)]
    if not selected_cols:
        raise ValueError(f"No columns found for wavelength {target} within tolerance {tol}")
    # Use Polars' mean_horizontal for row-wise mean
    return pl.mean_horizontal([pl.col(c) for c in selected_cols])

def index_calc(df):
    tol = 0.5  # Adjust tolerance as needed

    # Get mean expressions for each target wavelength
    w630 = get_wavelength_mean_expr(df, 630, tol)
    w680 = get_wavelength_mean_expr(df, 680, tol)
    w600 = get_wavelength_mean_expr(df, 600, tol)
    w525 = get_wavelength_mean_expr(df, 525, tol)
    w685 = get_wavelength_mean_expr(df, 685, tol)
    w440 = get_wavelength_mean_expr(df, 440, tol)
    w780 = get_wavelength_mean_expr(df, 780, tol)
    w550 = get_wavelength_mean_expr(df, 550, tol)
    w700 = get_wavelength_mean_expr(df, 700, tol)
    w740 = get_wavelength_mean_expr(df, 740, tol)
    w670 = get_wavelength_mean_expr(df, 670, tol)
    w730 = get_wavelength_mean_expr(df, 730, tol)
    w800 = get_wavelength_mean_expr(df, 800, tol)
    w470 = get_wavelength_mean_expr(df, 470, tol)
    w635 = get_wavelength_mean_expr(df, 635, tol)
    w415 = get_wavelength_mean_expr(df, 415, tol)
    w435 = get_wavelength_mean_expr(df, 435, tol)
    w531 = get_wavelength_mean_expr(df, 531, tol)
    w570 = get_wavelength_mean_expr(df, 570, tol)
    w678 = get_wavelength_mean_expr(df, 678, tol)
    w500 = get_wavelength_mean_expr(df, 500, tol)
    w760 = get_wavelength_mean_expr(df, 760, tol)
    w675 = get_wavelength_mean_expr(df, 675, tol)
    w650 = get_wavelength_mean_expr(df, 650, tol)
    w750 = get_wavelength_mean_expr(df, 750, tol)
    w510 = get_wavelength_mean_expr(df, 510, tol)
    w790 = get_wavelength_mean_expr(df, 790, tol)
    w720 = get_wavelength_mean_expr(df, 720, tol)
    w970 = get_wavelength_mean_expr(df, 970, tol)
    w900 = get_wavelength_mean_expr(df, 900, tol)

    # For RGR_Ratio, we use columns around 630 and 480 as an example,
    # but in your script it looks like you meant columns in [630..750] vs [480..560].
    # Adjust as appropriate to match your original design.
    rgr_numerator_cols = [
        c for c in df.columns if is_within_target(c, 630, tol) or is_within_target(c, 750, tol)
    ]
    rgr_denominator_cols = [
        c for c in df.columns if is_within_target(c, 480, tol) or is_within_target(c, 560, tol)
    ]

    # If you need a single average for 630..750, you can gather all columns in that range, etc.
    # For simplicity, let's do a straightforward approach with mean_horizontal.
    if rgr_numerator_cols:
        rgr_numerator = pl.mean_horizontal([pl.col(c) for c in rgr_numerator_cols])
    else:
        # fallback or raise error
        rgr_numerator = pl.lit(None)

    if rgr_denominator_cols:
        rgr_denominator = pl.mean_horizontal([pl.col(c) for c in rgr_denominator_cols])
    else:
        # fallback or raise error
        rgr_denominator = pl.lit(None)

    exprs = [
        (w630 / w680).alias("OCAR"),
        (w600 / w680).alias("YCAR"),
        (w525 / w685).alias("f525685"),
        (w440 / w685).alias("f440685"),
        (w780 / w550).alias("NIRGREEN"),
        (w780 / w700).alias("NIRRED"),
        (w780 / w740).alias("NIRNIR"),
        (w670 / w730).alias("f760730"),
        ((w800 - w670) / (w800 + w670 + 0.16)).alias("OSAVI"),
        (rgr_numerator / rgr_denominator).alias("RGR_Ratio"),
        ((w800 - w470) / (w800 + w470)).alias("PSNDc"),
        ((w800 - w635) / (w800 + w635)).alias("PSNDb"),
        ((w800 - w680) / (w800 + w680)).alias("PSNDa"),
        ((w780 - w550) / (w780 + w550)).alias("GNDVI"),
        ((w780 - w670) / (w780 + w670)).alias("RNDVI"),
        ((w800 - w435) / (w800 + w435)).alias("SIPI"),
        ((w970 - w900) / (w970 + w900)).alias("NWI"),
        (w970 / w900).alias("WI"),
        (w800 / w470).alias("PSSRc"),
        (w800 / w635).alias("PSSRb"),
        (w800 / w680).alias("PSSRa"),
        ((w415 - w435) / (w415 + w435)).alias("NPQ1"),
        ((w531 - w570) / (w531 + w570)).alias("PSSRaPRI"),
        ((w678 - w500) / w750).alias("PSRI"),
        (w760 / w500).alias("RARSc"),
        (w675 / (w650 * w700)).alias("RARSb"),
        (w675 / w700).alias("RARSa"),
        (((1 / w550) - (1 / w700))).alias("ARI"),
        (((1 / w510) - (1 / w700))).alias("CRI2"),
        (((1 / w510) - (1 / w550))).alias("CRI1"),
        (w900 / w510).alias("GDVI"),
        (w900 / w680).alias("SR"),
        ((w790 - w720) / (w790 + w720)).alias("NDRE"),
        ((w900 - w680) / (w900 + w680)).alias("NDVI"),
    ]
    return df.with_columns(exprs)

def preprocess(zip_path=None):
    zip_file = ensure_zip_file(zip_path)
    os.makedirs("./results", exist_ok=True)
    simple_logger("Starting extraction process...")
    extractor(zip_file)
    simple_logger("Creating row DataFrame...")
    row_df = creat_row_data_fram()
    simple_logger("Calculating relative values...")
    relative_df = cal_relative(row_df)
    simple_logger("Grouping and calculating mean values...")
    grouped_df = group_and_calculate_mean(relative_df)
    simple_logger("Calculating spectral indices...")
    grouped_df = index_calc(grouped_df)
    output_csv = "./results/relativ_with_indices.csv"
    grouped_df.write_csv(output_csv)
    simple_logger(f"Grouped DataFrame with indices saved to {output_csv}")

if __name__ == "__main__":
    os.makedirs("./results", exist_ok=True)
    zip_file = ensure_zip_file()
    simple_logger("Starting extraction process (standalone run)...")
    extractor(zip_file)
    simple_logger("Creating row DataFrame...")
    row_df = creat_row_data_fram()
    simple_logger("Calculating relative values...")
    relative_df = cal_relative(row_df)
    simple_logger("Grouping and calculating mean values...")
    grouped_df = group_and_calculate_mean(relative_df)
    simple_logger("Calculating spectral indices...")
    grouped_df = index_calc(grouped_df)
    output_csv = "./results/relativ_with_indices.csv"
    grouped_df.write_csv(output_csv)
    simple_logger(f"Grouped DataFrame with indices saved to {output_csv}")
