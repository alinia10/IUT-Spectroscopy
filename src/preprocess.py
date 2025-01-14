#!/usr/bin/env python3

import re
import polars as pl
import os 
from zipfile import ZipFile

from conf import config
from log import simple_logger

path_to_zip_file = f"./DATA/{config["inputs"]["zip_name"]}.zip"
directory_to_extract_to = f"./results/"

def extractor():

    with ZipFile(path_to_zip_file, 'r') as zip_ref:
        simple_logger(f"start extraxting {config["inputs"]["zip_name"]} to results folder ... ")

        zip_ref.extractall(directory_to_extract_to)

        simple_logger(f"extract completed.")


if __name__ =="__main__":
    extractor()