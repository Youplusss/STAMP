import os
import argparse

from lib.cli import add_common_args, finalize_args
from lib.paths import resolve_dataset_paths

from lib.evaluate import *

parser = argparse.ArgumentParser(description='PyTorch Prediction Model on Time-series Dataset')
add_common_args(parser)
args = finalize_args(parser.parse_args())

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

from model.net import *
from trainer import Tester
from lib.logger import get_logger
from lib.dataloader_smd import load_data3, load_data_unsup_train
from lib.dataloader_swat import load_data as swat_load_data, load_data2 as swat_load_data2
from lib.dataloader_wadi import load_data as wadi_load_data, load_data2 as wadi_load_data2
from lib.utils import *
from model.utils import *

DEVICE = get_default_device()
args.device = DEVICE

paths = resolve_dataset_paths(
    args.data,
    data_root=args.data_root,
    dataset_root=args.dataset_root,
    group_name=args.group_name,
    train_file=args.train_file,
    test_file=args.test_file,
    unsup_npz=args.unsup_npz,
)
