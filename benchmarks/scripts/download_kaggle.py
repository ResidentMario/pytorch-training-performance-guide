#!/usr/bin/env python3
"""
This CLI can be used to download one or more of the sample datasets we use for benchmarking from
Kaggle.

Example usage:

> ./download_kaggle.py tweet-sentiment-extraction
"""

import argparse
import os
import shutil
import subprocess
import pathlib
import zipfile


DATASET_NAMES = {"abhishek/bert-base-uncased"}
COMPETITION_DATASET_NAMES = {"tweet-sentiment-extraction"}


def main():
    parser = argparse.ArgumentParser(
        description="Download one or more benchmark datasets to local disk."
    )
    parser.add_argument(
        "names", type=str, nargs="+", help="Names of the datasets to download"
    )
    parser.add_argument(
        "-d", "--dest", type=str, help="Destination to download datasets to"
    )
    args = parser.parse_args()

    validate_names(args.names)
    validate_dest(args.dest)
    validate_envvars()
    validate_kaggle_install()

    dest = (
        pathlib.Path(args.dest).resolve()
        if args.dest is not None
        else pathlib.Path(".").resolve()
    )
    for name in args.names:
        kaggle_type = "datasets" if name in DATASET_NAMES else "competitions"
        download_dataset(name, kaggle_type, dest)
        if kaggle_type == "competitions":
            unpack_zipfile(dest / name, name + ".zip")


def validate_names(names: list) -> None:
    "Checks that the dataset name being passed in the list of known datasets."
    for name in names:
        if name not in DATASET_NAMES and name not in COMPETITION_DATASET_NAMES:
            raise ValueError(
                f"Got unexpected dataset '{name}', must be one of {DATASET_NAMES} "
                f"or {COMPETITION_DATASET_NAMES}"
            )


def validate_dest(dest: str | None) -> None:
    "Checks that the destination is valid (is a valid path and is not a file that already exists)."
    if dest is not None and pathlib.Path(dest).resolve().is_file():
        raise ValueError(f"Destination '{dest}' is a file that already exists.")


def validate_envvars() -> None:
    "Checks that the required KAGGLE_USERNAME and KAGGLE_KEY environmen variables are set."
    if "KAGGLE_USERNAME" not in os.environ:
        raise ValueError("Missing the required 'KAGGLE_USERNAME' environment variable.")
    if "KAGGLE_KEY" not in os.environ:
        raise ValueError(
            "Missing the required 'KAGGLE_KEY' environment variable. This should be parameterized "
            "with a valid Kaggle API key."
        )


def validate_kaggle_install() -> None:
    "Checks that the `kaggle` package is installed and available on PATH."
    try:
        import kaggle  # noqa: F401
    except ImportError:
        raise ImportError("The 'kaggle' Python package is not installed.")
    if shutil.which("kaggle") is None:
        raise ValueError("The 'kaggle' executable could not be found.")


def download_dataset(dataset_name: str, kaggle_type: str, dest: pathlib.Path) -> None:
    """
    Downloads a Kaggle dataset to disk, shelling out to the 'kaggle' CLI to do so.

    Successfully running this command requires installing the 'kaggle' PyPi package; having the
    CLI entrypoint that comes with it available on your PATH; and having the KAGGLE_USERNAME and
    KAGGLE_KEY environment variables set. Refer to https://github.com/Kaggle/kaggle-api for more
    information.

    Kaggle datasets come in two flavors: "datasets" that are uploaded directly to the platform by
    end users (or in rare case Kaggle staff), and "competitions" that are uploaded to the platform
    as part of a Kaggle competition. You will not be able to access competition data without first
    accepting the competition rules (this is even true of old competitions that finished running
    long ago) on the web by visiting https://www.kaggle.com/c/<competition-name>/rules.
    """
    kaggle_types = {"datasets", "competitions"}
    if kaggle_type not in kaggle_types:
        raise ValueError(
            f"Got unexpected kaggle_type '{kaggle_type}', must be one of {kaggle_types}"
        )
    dest = (
        dest / dataset_name.split("/")[-1]
    )  # remove the UPLOADERNAME/ prefix that datasets have
    dest.mkdir(exist_ok=True, parents=True)
    cmd = ["kaggle", kaggle_type, "download", dataset_name, "--path", dest.as_posix()]
    # NOTE(aleksey): the `kaggle datasets download` command has an `unzip` argument that unzips the
    # file in place. The `kaggle competitions download` command lacks this parameter. This is
    # because competitions, a much older feature of the platform, do not universially use a
    # zipfile. The upshot is that we have to uncompress any competition data we use ourselves.
    if kaggle_type == "datasets":
        cmd.append("--unzip")

    subprocess.run(cmd, check=True)


def unpack_zipfile(root_dir: pathlib.Path, filename: str) -> None:
    """
    Unzips a zipfie. Only needed for competition datasets; regular datasets are unzipped using the
    `unzip` parameter packaged with the `kaggle` CLI. See the dev note in `download_dataset` for
    why this is necessary.
    """
    zf_path = root_dir / filename
    zf = zipfile.ZipFile(zf_path)
    zf.extractall(root_dir)
    pathlib.Path(zf_path).unlink()


if __name__ == "__main__":
    main()
