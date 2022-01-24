import argparse
import os
import pprint
from datetime import datetime
from pathlib import Path

import yaml

from data_loading.cots_data_splitter import COTSDataSplitter
from data_loading.load_data import create_cots_dataloader
from models.models import load_darknet_model
from training.train_model_builder import TrainModelBuilder
from training.trainer import Trainer
from logger import colorstr, save_logs
from logger import logger
from utils.constants import TRAIN_SUBDIR
from utils.general import increment_path
from models.model_manager import YOLOModelManager

LOCAL_RANK = int(
    os.getenv("LOCAL_RANK", -1)
)  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))


def get_parser() -> argparse.Namespace:
    """Get argument parser
    Modify this function as your project needs
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        help="Model " + colorstr("weight") + "  file path",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help=colorstr("Dataset config") + " file path",
    )
    parser.add_argument(
        "--training-cfg",
        type=str,
        default=os.path.join("res", "configs", "cfg", "train_config.yaml"),
        help=colorstr("Training config") + " file path",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="DDP parameter. " + colorstr("red", "bold", "Do not modify"),
    )
    parser.add_argument("--log-dir", type=str, default="", help="Log root directory.")
    parser.add_argument(
        "--use-swa",
        action="store_true",
        default=False,
        help="Apply SWA (Stochastic Weight Averaging) or not",
    )
    parser.add_argument(
        "--debug-imp",
        action='store_true',
        help="Only used to debug code implementation")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_parser()

    # Load training data config
    with open(args.data, "r") as f:
        data_cfg = yaml.safe_load(f)

    # Load training hyperparameters
    with open(args.training_cfg, "r") as f:
        train_cfg = yaml.safe_load(f)

    model_chkpnt = args.model_checkpoint

    # Set log directory
    if args.log_dir:
        train_cfg["train"]["log_dir"] = args.log_dir
    base_log_dir = train_cfg["train"]["log_dir"] or "exp"
    log_dir = str(
        increment_path(
            path_=str(Path(base_log_dir) / TRAIN_SUBDIR / datetime.now().strftime("%Y_%m%d_runs")),
            mkdir=True))

    train_cfg["train"]["log_dir"] = log_dir
    train_cfg["train"]["weights"] = model_chkpnt

    # All configs
    cfg_all = {
        "data_cfg": data_cfg,
        "train_cfg": train_cfg,
        "args": vars(args),
    }

    if train_cfg['train']['save_logs']:
        save_logs(filepath=Path(log_dir) / "logs.txt")

    logger.info(
        "\n"
        + colorstr("red", "bold", f"{'-' * 30} Training Configs START {'-' * 30}")
        + "\n"
        + pprint.pformat(cfg_all, indent=4)
        + "\n"
        + colorstr("red", "bold", f"{'-' * 30} Training Configs END {'-' * 30}")
    )

    model = load_darknet_model(
        model_cfg_path=train_cfg['train']['cfg'],
        weight_path=model_chkpnt,
        load_ema=True
    )

    # Set training model builder
    train_builder = TrainModelBuilder(model, train_cfg, log_dir, full_cfg=cfg_all)
    train_builder.ddp_init()

    # Fetch stride size from the model
    stride_size = int(model.module_list[-1].stride)  # type: ignore

    # Data splitter
    data_splitter = COTSDataSplitter(
        train_perc=data_cfg['train_perc'],
        num_groups=data_cfg['num_splits'],
        csv_file=data_cfg['data_csv'])

    # Only for debugging
    if args.debug_imp:
        data_splitter.reduce_df_sizes()  # reduce dataframe sizes
        cfg_all['train_cfg']['train']['workers'] = 0  # Set workers = 0 to debug a single process

    # Set data loader and data set
    train_loader, train_dataset = create_cots_dataloader(
        video_path=data_cfg['video_path'],
        dataframe=data_splitter.train_df,
        cfg=train_cfg,
        stride=stride_size,
        prefix="[Train] "
    )

    # Set data loader and data set
    if RANK in [-1, 0]:
        val_loader, val_dataset = create_cots_dataloader(
            video_path=data_cfg['video_path'],
            dataframe=data_splitter.val_df,
            cfg=train_cfg,
            stride=stride_size,
            prefix="[Val] ",
            validation=True,
        )
    else:
        val_loader, val_dataset = None, None

    # Manage model properties
    model_manager = YOLOModelManager(model, train_cfg, train_builder.device, train_builder.wdir)
    model_manager.freeze(train_cfg["train"]["freeze"])
    model, ema, device = train_builder.prepare()
    model_manager.model = model
    model = model_manager.set_model_params(train_dataset, ema=ema)

    # Set model trainer
    trainer = Trainer(
        model=model,
        cfg=train_cfg,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        ema=ema,
        device=train_builder.device,
        log_dir=train_builder.log_dir,
        use_swa=args.use_swa,
    )

    # Start training
    trainer.train(start_epoch=model_manager.start_epoch)
