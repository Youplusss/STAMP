import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DatasetPaths:
    """Resolved file paths for a dataset.

    Some datasets use CSV (SWaT/WADI), some use PKL (SMD/MSL/SMAP), and some use NPZ (unsupervised sets).
    Not every field is meaningful for every dataset.
    """

    dataset: str
    base_dir: str
    data_root: str
    dataset_root: str

    # CSV-style datasets (SWaT/WADI)
    train_csv: Optional[str] = None
    test_csv: Optional[str] = None

    # SMD subset name (machine-1-1 etc.)
    group_name: Optional[str] = None

    # Unsupervised npz
    unsup_npz: Optional[str] = None


def get_base_dir() -> str:
    # Always resolve relative to repo root (the directory containing this file is lib/)
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def resolve_dataset_paths(
    dataset: str,
    *,
    base_dir: Optional[str] = None,
    data_root: Optional[str] = None,
    dataset_root: Optional[str] = None,
    group_name: Optional[str] = None,
    train_file: Optional[str] = None,
    test_file: Optional[str] = None,
    unsup_npz: Optional[str] = None,
) -> DatasetPaths:
    """Resolve dataset-related paths in a consistent repo-wide way."""

    base_dir = base_dir or get_base_dir()
    data_root = data_root or os.path.join(base_dir, "data")
    dataset_root = dataset_root or os.path.join(base_dir, "dataset")

    ds = dataset
    ds_lower = ds.lower()

    # Defaults
    train_csv = train_file
    test_csv = test_file

    if ds_lower == "swat":
        train_csv = train_csv or os.path.join(dataset_root, ds, "swat_train.csv")
        test_csv = test_csv or os.path.join(dataset_root, ds, "swat_test.csv")
        unsup_npz = unsup_npz or os.path.join(data_root, "unsupervised_data", "test_data_swat_unsup.npz")

    elif ds_lower == "wadi":
        train_csv = train_csv or os.path.join(dataset_root, ds, "wadi_train.csv")
        test_csv = test_csv or os.path.join(dataset_root, ds, "wadi_test.csv")
        unsup_npz = unsup_npz or os.path.join(data_root, "unsupervised_data", "test_data_wadi_unsup.npz")

    elif ds_lower == "smd":
        group_name = group_name or "machine-1-1"
        unsup_npz = unsup_npz or os.path.join(data_root, "unsupervised_data", "test_data_smd_unsup.npz")

    elif ds_lower in {"msl", "smap"}:
        unsup_npz = unsup_npz or os.path.join(data_root, "unsupervised_data", f"test_data_{ds_lower}_unsup.npz")

    else:
        # Still return something, but caller may handle dataset-specific logic.
        unsup_npz = unsup_npz or os.path.join(data_root, "unsupervised_data", f"test_data_{ds}_unsup.npz")

    return DatasetPaths(
        dataset=ds,
        base_dir=base_dir,
        data_root=data_root,
        dataset_root=dataset_root,
        train_csv=train_csv,
        test_csv=test_csv,
        group_name=group_name,
        unsup_npz=unsup_npz,
    )

