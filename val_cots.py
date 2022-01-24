import argparse
import os
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml
from torch import nn

from data_loading.augmentation.cots_data_loader import LoadCOTSImagesAndLabels
from data_loading.cots_data_splitter import COTSDataSplitter
from logger import colorstr, get_logger
from models.models import load_darknet_model
from training.train_model_builder import TrainModelBuilder
from utils.constants import VAL_SUBDIR
from utils.general import increment_path
from models.model_manager import YOLOModelManager
from utils.torch_utils import count_param
from training.validator import YoloValidator

LOGGER = get_logger(__name__)


def get_parser() -> argparse.Namespace:
    """Get argument parser."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--weights", type=str, default="", help="Model weight path.")
    parser.add_argument(
        "--model-cfg", type=str, default="", help="Model config file path."
    )
    parser.add_argument(
        "--data-cfg",
        type=str,
        default="res/configs/data/coco.yaml",
        help="Validation image root.",
    )
    parser.add_argument(
        "--training-cfg",
        type=str,
        default=os.path.join("res", "configs", "cfg", "train_config.yaml"),
        help=colorstr("Training config") + " file path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="GPU device id. '' will use all GPUs. EX) '0,2' or 'cpu'",
    )
    parser.add_argument(
        "--dst",
        type=str,
        default="exp",
        help="Export directory. Directory will be {dst}/val/{DATE}_runs1, ...",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("-iw", "--img-width", type=int, default=640, help="Image width")
    parser.add_argument(
        "-ih",
        "--img-height",
        type=int,
        default=-1,
        help="Image height. (-1 will set image height to be identical to image width.)",
    )
    parser.add_argument(
        "-ct", "--conf-t", type=float, default=0.001, help="Confidence threshold."
    )
    parser.add_argument(
        "-it", "--iou-t", type=float, default=0.65, help="IoU threshold."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=512,
        help="Use top-k objects in NMS layer (TensorRT only)",
    )
    parser.add_argument(
        "-ktk",
        "--keep-top-k",
        default=100,
        help="Keep top-k after NMS. This must be less or equal to top-k (TensorRT only)",
    )
    parser.add_argument(
        "--rect",
        action="store_true",
        dest="rect",
        default=True,
        help="Use rectangular image",
    )
    parser.add_argument(
        "--no-rect", action="store_false", dest="rect", help="Use squared image.",
    )
    parser.add_argument(
        "--single-cls",
        action="store_true",
        default=False,
        help="Validate as single class only.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=False,
        help="Save validation result plot.",
    )
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")
    parser.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help="Run profiling before validation.",
    )
    parser.add_argument(
        "--n-profile",
        type=int,
        default=100,
        help="Number of n iteration for profiling.",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        default=False,
        help="Run half precision model (PyTorch only)",
    )
    parser.add_argument(
        "--hybrid-label",
        action="store_true",
        default=False,
        help="Run NMS with hybrid information (ground truth label + predicted result.) "
             "(PyTorch only) This is for auto-labeling purpose.",
    )
    parser.add_argument(
        "--nms_type",
        type=str,
        default="nms",
        help="NMS type (e.g. nms, batched_nms, fast_nms, matrix_nms, merge_nms)",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        default=False,
        help="Apply TTA (Test Time Augmentation)",
    )
    parser.add_argument(
        "--tta-cfg",
        type=str,
        default="res/configs/cfg/tta.yaml",
        help="TTA config file path",
    )
    parser.add_argument(
        "--n-skip", type=int, default=0, help="n skip option for data loader."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_parser()

    # Either weights or model_cfg must be provided.
    if args.weights == "" and args.model_cfg == "":
        LOGGER.error(
            "Either "
            + colorstr("bold", "--weight")
            + " or "
            + colorstr("bold", "--model-cfg")
            + " must be provided."
        )
        exit(1)

    # Parse input files
    with open(args.data_cfg, "r") as f:
        data_cfg = yaml.safe_load(f)

    with open(args.training_cfg, "r") as f:
        train_cfg = yaml.safe_load(f)
    hyp = train_cfg['hyper_params']

    # Set and create session directory
    base_log_dir = train_cfg["train"]["log_dir"] or "exp"
    log_dir = str(
        increment_path(
            path_=str(Path(base_log_dir) / VAL_SUBDIR / datetime.now().strftime("%Y_%m%d_runs")),
            mkdir=True))

    # Set configurations
    cfg_all = {
        "data_cfg": data_cfg,
        "train_cfg": train_cfg,
        "args": vars(args),
    }

    # Load model
    model = load_darknet_model(weight_path=args.weights, model_cfg_path=args.model_cfg, load_ema=True)
    train_builder = TrainModelBuilder(model, train_cfg, log_dir, full_cfg=cfg_all)
    train_builder.ddp_init()
    stride_size = int(model.module_list[-1].stride)

    # Set validation config
    val_cfg = {
        "train": {
            "single_cls": args.single_cls,
            "plot": args.plot,
            "batch_size": args.batch_size,
            "image_size": train_cfg['train']['image_size'],
        },
        "hyper_params": {"conf_t": train_cfg['hyper_params']['conf_t'], "iou_t": train_cfg['hyper_params']['iou_t']},
    }

    # Set data split
    # TODO: more options for dataset to validate
    data_splitter = COTSDataSplitter(
        train_perc=data_cfg['train_perc'],
        num_groups=data_cfg['num_splits'],
        csv_file=data_cfg['data_csv'])

    val_dataset = LoadCOTSImagesAndLabels(
        video_path=data_cfg['video_path'],
        dataframe=data_splitter.val_df,
        img_size=train_cfg['train']['image_size'],
        batch_size=args.batch_size,
        stride=stride_size,
    )

    # Set validation data loader
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=min(os.cpu_count(), args.batch_size),  # type: ignore
        pin_memory=True,
        collate_fn=LoadCOTSImagesAndLabels.collate_fn,
    )

    # Set model parameters
    model_manager = YOLOModelManager(model, train_cfg, train_builder.device, train_builder.wdir)
    model_manager.freeze(train_cfg["train"]["freeze"])

    # Get updated model and device
    model, ema, device = train_builder.prepare()
    model_manager.model = model
    model = model_manager.set_model_params(val_dataset, ema=ema)

    if isinstance(model, torch.jit.ScriptModule):
        model.to(device).eval()
    elif isinstance(model, nn.Module):
        model.to(device).fuse().eval()  # type: ignore
        LOGGER.info(f"# of parameters: {count_param(model):,d}")
        if args.half:
            model.half()

    # Set model validator
    validator = YoloValidator(
        model,
        val_loader,
        device,
        val_cfg,
        compute_loss=True,
        hybrid_label=args.hybrid_label,
        half=args.half,
        log_dir=args.dst,
        incremental_log_dir=True,
        export=True,
        nms_type=args.nms_type,
        tta=args.tta,
    )
    t0 = time.monotonic()
    val_result = validator.validation()
    time_took = time.monotonic() - t0

    LOGGER.info(f"Time took: {time_took:.5f}s")

    with open(os.path.join(validator.log_dir, "args.yaml"), "w") as f:
        yaml.dump(vars(args), f)
