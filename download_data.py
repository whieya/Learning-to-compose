"""
Utility script to download datasets.
Code from https://github.com/addtt/object-centric-library/blob/main/download_data.py.
Use this script to download clevrtex dataset in hdf5 format.
"""

import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
from typing import List

from data import get_available_dataset_configs
from utils.utils import download_file

GLOBAL = ["README.txt", "LICENSE"]

REMOTE_ROOT = "https://data-download.compute.dtu.dk/multi_object_datasets/"


def _dataset_files(name: str, *, include_style: bool) -> List[str]:
    extended_name = {
        "clevr": "clevr_10",
        "multidsprites": "multidsprites_colored_on_grayscale",
        "objects_room": "objects_room_train",
    }.get(name, name)
    datasets_without_style = ["clevrtex"]
    out = [f"{extended_name}-{suffix}" for suffix in ["full.hdf5", "metadata.npy"]]
    if include_style and name not in datasets_without_style:
        out.append(f"{name}-style.hdf5")
    return out


def _get_remote_address(name: str) -> str:
    assert REMOTE_ROOT.endswith("/")
    return REMOTE_ROOT + name


def _get_destination(path_dir:str, name: str) -> str:
    return str(Path(path_dir) / name)


if __name__ == "__main__":

    available_datasets = get_available_dataset_configs()

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-p",
        "--path_dir",
        required=True,
        help='paths to download the data'
        )

    parser.add_argument(
        "-d",
        "--datasets",
        nargs="+",
        required=True,
        help="Names of the datasets to be downloaded (space-separated list). If the "
        "string 'all' is given, all available datasets will be downloaded.",
    )
    parser.add_argument(
        "--include-style-transfer",
        action="store_true",
        help="Whether style transfer versions of the datasets should be downloaded too",
    )
    args = parser.parse_args()

    # Validate datasets and skip invalid ones.
    datasets = args.datasets
    if datasets == ["all"]:
        datasets = available_datasets
    else:
        missing = [d for d in datasets if d not in available_datasets]
        if missing:
            print(f"The following datasets are not valid: {missing}")
            datasets = [d for d in datasets if d in available_datasets]
    print(f"The following datasets will be downloaded: {datasets}")

    # Create data folder.
    Path(args.path_dir).mkdir(exist_ok=True)

    print(f"\nDownloading global files...")
    for filename in GLOBAL:
        download_file(_get_remote_address(filename), _get_destination(args.path_dir, filename))

    for dataset in datasets:
        print(f"\nDownloading files for '{dataset}'...")
        filenames = _dataset_files(dataset, include_style=args.include_style_transfer)
        for filename in filenames:
            destination = _get_destination(args.path_dir, filename)
            if Path(destination).exists():
                print(f"Destination file {destination} exists: skipping.")
                continue
            download_file(_get_remote_address(filename), destination)
