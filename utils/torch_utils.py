import math
import os
import random
import time
from contextlib import contextmanager
from copy import deepcopy
from typing import Generator, Optional, Union, List, Tuple, Dict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as functional

from logger import logger


def count_param(model: nn.Module) -> int:
    """Count number of all parameters
    :param model: PyTorch model
    :return: Sum of # of parameters
    """
    return sum(list(x.numel() for x in model.parameters()))


@contextmanager
def torch_distributed_zero_first(local_rank: int) -> Generator:
    """Make sure torch distributed call is run on only local_rank -1 or 0
    Decorator to make all processes in distributed training wait for each local_master
    to do something.
    """
    if local_rank not in [-1, 0]:
        dist.barrier(device_ids=[local_rank])  # type: ignore
    yield
    if local_rank == 0:
        dist.barrier(device_ids=[0])  # type: ignore


def init_torch_seeds(seed: int = 0) -> None:
    """Set random seed for torch
    If seed == 0, it can be slower but more reproducible
    If not, it would be faster but less reproducible
    Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    """
    torch.manual_seed(seed)

    if seed == 0:
        cudnn.deterministic = True
        cudnn.benchmark = False

    else:
        cudnn.deterministic = False
        cudnn.benchmark = True


def select_device(device: str = "", batch_size: Optional[int] = None) -> torch.device:
    """Select torch device
    :param device: 'cpu' or '0' or '0, 1, 2, 3' format string
    :param batch_size: distribute batch to multiple gpus
    :returns: a torch device
    """
    cpu_request = device.lower() == "cpu"
    if device and not cpu_request:
        os.environ["CUDA_VISIBLE_DEVICES"] = device
        assert torch.cuda.is_available(), (
                "CUDA unavailable, invalid device %s requested" % device
        )

    cuda = False if cpu_request else torch.cuda.is_available()
    if cuda:
        c = 1024 ** 2
        ng = torch.cuda.device_count()
        if ng > 1 and batch_size:
            assert (
                    batch_size % ng == 0
            ), "batch-size %g not multiple of GPU count %g" % (batch_size, ng)
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        s = "Using CUDA "
        for i in range(0, ng):
            if i == 1:
                s = " " * len(s)
                logger.info(
                    "%sdevice%g _CudaDeviceProperties(name='%s', total_memory=%dMB)"
                    % (s, i, x[i].name, x[i].total_memory / c)
                )

    else:
        logger.info("Using CPU")

    logger.info("")
    return torch.device("cuda:0" if cuda else "cpu")


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def is_parallel(model: nn.Module) -> bool:
    """Check if the model is DP or DDP
    :param model: PyTorch nn.Module
    :return: True if the model is DP or DDP, False otherwise
    """
    return type(model) in (
        nn.parallel.DataParallel,
        nn.parallel.DistributedDataParallel,
    )


def de_parallel(model: nn.Module) -> nn.Module:
    """Decapsule parallelized model.
    :param model: Single-GPU model, DP model or DDP model
    :return: a decapsulized single-GPU model
    """
    return model.module if is_parallel(model) else model  # type: ignore


def init_seeds(seed: int = 0) -> None:
    """Initialize random seeds"""
    random.seed(seed)
    np.random.seed(seed)
    init_torch_seeds(seed)


def intersect_dicts(
        da: dict, db: dict, exclude: Union[List[str], Tuple[str, ...]] = ()
) -> dict:
    """Check dictionary intersection of matching keys and shapes
    Omitting 'exclude' keys, using da values.
    """
    return {
        k: v
        for k, v in da.items()
        if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape
    }


def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True


def find_modules(model, mclass=nn.Conv2d):
    # Finds layer indices matching module class 'mclass'
    return [i for i, m in enumerate(model.module_list) if isinstance(m, mclass)]


def load_model_weights(
        model: nn.Module, weights: Union[Dict, str], exclude: Optional[list] = None,
) -> nn.Module:
    """Load model's pretrained weights
    :param model: model instance to load weight
    :param weights: model weight path
    :param exclude: exclude list of layer names
    :return: self.model which the weights has been loaded
    """
    if isinstance(weights, str):
        ckpt = torch.load(weights)
    else:
        ckpt = weights

    exclude_list = [] if exclude is None else exclude

    state_dict = ckpt["model"].float().state_dict()
    state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude_list)
    model.load_state_dict(state_dict, strict=False)  # load weights
    logger.info(
        "Transferred %g/%g items from %s"
        % (
            len(state_dict),
            len(model.state_dict()),
            weights if isinstance(weights, str) else weights.keys(),
        )
    )
    return model


def sparsity(model: nn.Module) -> float:
    """Compute global model sparsity
    :param model: PyTorch model
    :return: sparsity ratio (sum of zeros / # of parameters)
    """
    n_param, zero_param = 0.0, 0.0
    for p in model.parameters():
        n_param += p.numel()
        zero_param += (p == 0).sum()  # type: ignore
    return zero_param / n_param


def fuse_conv_and_bn(conv, bn):
    # Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.size()))

    # prepare spatial bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


def model_info(model, verbose=False, img_size=640):
    # Model information. img_size may be int or list, i.e. img_size=640 or img_size=[640, 320]
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if verbose:
        print('%5s %40s %9s %12s %20s %10s %10s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))

    try:  # FLOPS
        from thop import profile
        flops = profile(deepcopy(model), inputs=(torch.zeros(1, 3, img_size, img_size),), verbose=False)[0] / 1E9 * 2
        fs = ', %.9f GFLOPS' % flops  # 640x640 FLOPS
    except (ImportError, Exception):
        fs = ''

    logger.info(f"Model Summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}")


def scale_img(
        img: torch.Tensor, ratio: float = 1.0, same_shape: bool = False, gs: int = 32
) -> torch.Tensor:
    """Scales img(bs,3,y,x) by ratio constrained to gs-multiple.
    Reference: https://github.com/ultralytics/yolov5/blob/master/utils/torch_utils.py#L257-L267
    :param img: image tensor
    :param ratio: scale ratio for image tensor
    :param same_shape: whether to make same shape or not
    :param gs: stride
    :returns: scaled image tensor
    """
    if ratio == 1.0:
        return img
    else:
        h, w = img.shape[2:]
        s = (int(h * ratio), int(w * ratio))  # new size
        img = functional.interpolate(img, size=s, mode="bilinear", align_corners=False)  # resize
        if not same_shape:  # pad/crop img
            h, w = (math.ceil(x * ratio / gs) * gs for x in (h, w))
        return functional.pad(
            img, [0, w - s[1], 0, h - s[0]], value=0.447
        )  # value = imagenet mean


def copy_attr(
        a: object,
        b: object,
        include: Union[List[str], Tuple[str, ...]] = (),
        exclude: Union[List[str], Tuple[str, ...]] = (),
) -> None:
    """Copy attributes from b to a, options to only include and to exclude.
    :param a: destination
    :param b: source
    :param include: key names to copy
    :param exclude: key names NOT to copy
    """
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith("_") or k in exclude:
            continue
        else:
            setattr(a, k, v)


class ModelEMA:
    """Model Exponential Moving Average.

    from https://github.com/rwightman/pytorch-image-
    models Keep a moving average of everything in the model state_dict (parameters and
    buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(
            self, model: nn.Module, decay: float = 0.9999, updates: int = 0
    ) -> None:
        """Initialize ModelEMA class."""
        # Create EMA
        self.ema = deepcopy(model).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (
                1 - math.exp(-x / 2000)
        )  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model: nn.Module) -> None:
        """Update EMA parameters."""
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    key = k if k in msd else f"module.{k}"
                    v *= d
                    v += (1.0 - d) * msd[key].detach()

    def update_attr(
            self,
            model: nn.Module,
            include: Union[List[str], Tuple[str, ...]] = (),
            exclude: tuple = ("process_group", "reducer"),
    ) -> None:
        """Update EMA attributes."""
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)
