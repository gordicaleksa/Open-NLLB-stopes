#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from glob import glob
from pathlib import Path
import tarfile
from typing import Dict


import requests
import yaml


def download_flores200(flores_path):
    corpus_name = "flores200"
    download_url = (
    "https://tinyurl.com/flores200dataset"
    )
    response = requests.get(download_url)
    if not response.ok:
        raise Exception(f"Could not download from {download_url}!")
    download_path = os.path.join(flores_path.parent, f"{corpus_name}.tar.gz")
    open(download_path, "wb").write(response.content)
    print(f"Wrote: {download_path}")

    with tarfile.open(download_path) as tar:
        tar.extractall(flores_path)
    os.remove(download_path)


def get_scaling_factors(flores_root_path):
    flores_path = Path(flores_root_path / "flores200")
    if not flores_path.exists():
        download_flores200(flores_path)

    devsets = glob(str(Path(flores_path) / "flores200_dataset" / "dev" / "*.dev"))
    factors: Dict[str, float] = {}  # lang -> factor
    for devset in devsets:
        _, fname = os.path.split(devset)
        lang = fname[:-4].replace("-", "_")
        with open(devset, "rt") as fin:
            factors[lang] = len(fin.read())

    # rescale everything based on English
    return {lang: factors["eng_Latn"] / lang_factor for lang, lang_factor in factors.items()}


def main(args):
    args.data_conf_dir.mkdir(parents=True, exist_ok=True)
    with open(args.data_conf_dir / "length_factors.yaml", "wt") as fout:
        yaml.safe_dump(get_scaling_factors(args.flores_root_path), fout)


if __name__ == "__main__":
    # You can just directly run this script and leave everything as default.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--flores-root-path",
        type=Path,
        default=Path("stopes/pipelines/filtering/"),
        help="Location of FLORES data used for computing factors.",
    )
    parser.add_argument(
        "--data-conf-dir",
        type=Path,
        default=Path("stopes/pipelines/filtering/filter_configs"),
        help="Directory where the configuration files are stored.",
    )
    args = parser.parse_args()
    main(args)
