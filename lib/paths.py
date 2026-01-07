import os
from dataclasses import dataclass
from typing import Optional
from datetime import datetime


@dataclass(frozen=True)
class DatasetPaths:
    dataset: str
    base_dir: str
    data_root: str
    dataset_root: str
    train_csv: Optional[str] = None
    test_csv: Optional[str] = None
    group_name: Optional[str] = None
    unsup_npz: Optional[str] = None


@dataclass(frozen=True)
class ExperimentDirs:
    """Standard output directories for a single run.

    layout:
      <root>/log  - text logs
      <root>/pth  - checkpoints
      <root>/pdf  - figures

    run_id is a short timestamp used to disambiguate log filenames.
    """

    root: str
    log_dir: str
    pth_dir: str
    pdf_dir: str
    run_id: str


def get_base_dir() -> str:
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

    base_dir = base_dir or get_base_dir()
    data_root = data_root or os.path.join(base_dir, "data")
    dataset_root = dataset_root or os.path.join(base_dir, "dataset")

    ds = dataset
    ds_lower = ds.lower()
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
        # Prefer per-machine files so --group works; fallback to concatenated smd_train/smd_test
        cand_train = os.path.join(dataset_root, ds, f"{group_name}_train.csv")
        cand_test = os.path.join(dataset_root, ds, f"{group_name}_test.csv")
        if train_csv is None and os.path.isfile(cand_train):
            train_csv = cand_train
        if test_csv is None and os.path.isfile(cand_test):
            test_csv = cand_test
        train_csv = train_csv or os.path.join(dataset_root, ds, "smd_train.csv")
        test_csv = test_csv or os.path.join(dataset_root, ds, "smd_test.csv")
        unsup_npz = unsup_npz or os.path.join(data_root, "unsupervised_data", "test_data_smd_unsup.npz")

    elif ds_lower == "msl":
        train_csv = train_csv or os.path.join(dataset_root, ds, "msl_train.csv")
        test_csv = test_csv or os.path.join(dataset_root, ds, "msl_test.csv")
        unsup_npz = unsup_npz or os.path.join(data_root, "unsupervised_data", "test_data_msl_unsup.npz")

    elif ds_lower == "smap":
        unsup_npz = unsup_npz or os.path.join(data_root, "unsupervised_data", "test_data_smap_unsup.npz")

    else:
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


def resolve_experiment_dirs(root: str, *, run_id: Optional[str] = None) -> ExperimentDirs:
    """Resolve and (if needed) create standard expe subdirectories.

    Args:
        root: usually 'expe'
        run_id: optional; default is YYYYmmdd_HHMMSS

    Returns:
        ExperimentDirs
    """
    root = root or 'expe'
    root = os.path.abspath(root)
    log_dir = os.path.join(root, 'log')
    pth_dir = os.path.join(root, 'pth')
    pdf_dir = os.path.join(root, 'pdf')

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(pth_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)

    rid = run_id or datetime.now().strftime('%Y%m%d_%H%M%S')

    return ExperimentDirs(
        root=root,
        log_dir=log_dir,
        pth_dir=pth_dir,
        pdf_dir=pdf_dir,
        run_id=rid,
    )
