from enum import Enum
from typing import List, Dict, Any, Set, Union, Iterable, Callable, Type

import torch
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from torch.optim.swa_utils import AveragedModel, SWALR
from yacs.config import CfgNode

from assr.model.common import Affine2d
from assr.utils.solver.lr_scheduler import WarmupMultiStepLR, WarmupCosineLR
from assr.utils.solver.optimizers import LARS, Lamb16, Lamb

"""
# 0. Gradient Clipping 梯度裁剪
"""

_GradientClipperInput = Union[torch.Tensor, Iterable[torch.Tensor]]
_GradientClipper = Callable[[_GradientClipperInput], None]


class GradientClipType(Enum):
    VALUE = "value"
    NORM = "norm"


def _create_gradient_clipper(cfg: CfgNode) -> _GradientClipper:
    """
    Creates gradient clipping closure to clip by value or by norm,
    according to the provided config.
    根据所提供的配置，创建梯度裁剪闭包，以按值或norm进行裁剪
    :param cfg:
    :return:
    """
    cfg = cfg.clone()

    def clip_grad_norm(p: _GradientClipperInput):
        torch.nn.utils.clip_grad_norm_(p, cfg.CLIP_VALUE, cfg.NORM_TYPE)

    def clip_grad_value(p: _GradientClipperInput):
        torch.nn.utils.clip_grad_value_(p, cfg.CLIP_VALUE)

    _GRADIENT_CLIP_TYPE_TO_CLIPPER = {
        GradientClipType.VALUE: clip_grad_value,
        GradientClipType.NORM: clip_grad_norm,
    }
    return _GRADIENT_CLIP_TYPE_TO_CLIPPER[GradientClipType(cfg.CLIP_TYPE)]


def _generate_optimizer_class_with_gradient_clipping(
        optimizer_type: Type[torch.optim.Optimizer],
        gradient_clipper: _GradientClipper) -> Type[torch.optim.Optimizer]:
    """
    Dynamically creates a new type that inherits the type of a given instance
    and overrides the `step` method to add gradient clipping
    动态创建一个继承给定实例类型的新类型，并覆盖“step”方法来添加梯度裁剪
    :param optimizer_type:
    :param gradient_clipper:
    :return:
    """

    def optimizer_wgc_step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                gradient_clipper(p)
        super(type(self), self).step(closure)

    OptimizerWithGradientClip = type(
        optimizer_type.__name__ + "WithGradientClip",
        (optimizer_type,),
        {"step": optimizer_wgc_step},
    )
    return OptimizerWithGradientClip


def maybe_add_gradient_clipping(
        cfg: CfgNode,
        optimizer: torch.optim.Optimizer) -> torch.optim.Optimizer:
    """
        If gradient clipping is enabled through config options, wraps the existing
        optimizer instance of some type OptimizerType to become an instance
        of the new dynamically created class OptimizerTypeWithGradientClip
        that inherits OptimizerType and overrides the `step` method to
        include gradient clipping.
        Args:
            cfg: CfgNode
                configuration options
            optimizer: torch.optim.Optimizer
                existing optimizer instance
        Return:
            optimizer: torch.optim.Optimizer
                either the unmodified optimizer instance (if gradient clipping is
                disabled), or the same instance with adjusted __class__ to override
                the `step` method and include gradient clipping
        """
    if not cfg.SOLVER.CLIP_GRADIENTS.ENABLED:
        return optimizer
    grad_clipper = _create_gradient_clipper(cfg.SOLVER.CLIP_GRADIENTS)
    OptimizerWithGradientClip = _generate_optimizer_class_with_gradient_clipping(
        type(optimizer), grad_clipper
    )
    optimizer.__class__ = OptimizerWithGradientClip
    return optimizer


"""
# 1. Build Optimizer
"""


def build_optimizer(cfg: CfgNode, model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Build an optimizer from config.
    :param cfg:
    :param model:
    :return: torch.optim.Optimizer
    """
    norm_module_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm,
        Affine2d,  # affine layer
    )

    res_name_list, res_factor = _create_residual_name(cfg)
    params: List[Dict[str, Any]] = []
    memo: Set[torch.nn.parameter.Parameter] = set()
    for module_name, module in model.named_modules():
        for key, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters 避免重复参数
            if value in memo:
                continue
            memo.add(value)
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if isinstance(module, norm_module_types):
                weight_decay = cfg.SOLVER.WEIGHT_DECAY_NORM
            elif key == "bias":
                lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
                weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
            # Scale learning rate for residual blocks 残差块的缩放学习率
            if res_factor is not None:
                prefix = module_name.split('.')[:6]
                prefix = '.'.join(prefix)
                if prefix in res_name_list:
                    lr *= res_factor

            params += [{
                "params": [value],
                "lr": lr,
                "weight_decay": weight_decay
            }]

    solver_name = cfg.SOLVER.NAME
    if solver_name == 'SGD':
        optimizer = torch.optim.SGD(
            params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM)
    elif solver_name in ['Adam', 'AdamW']:
        optimizer = getattr(torch.optim, solver_name)(
            params, cfg.SOLVER.BASE_LR, betas=cfg.SOLVER.BETAS)
    elif solver_name == 'LARS':  # Large-batch version of SGD
        optimizer = LARS(params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM)
    elif solver_name == 'Lamb':  # Large-batch version of Adam
        if cfg.MODEL.MIXED_PRECISION:
            optimizer = Lamb16(
                params, cfg.SOLVER.BASE_LR, betas=cfg.SOLVER.BETAS,
                eps=cfg.SOLVER.EPS, clamp_trust_ratio=cfg.SOLVER.CLAMP_TRUST_RATIO,
            )
        else:
            optimizer = Lamb(params, cfg.SOLVER.BASE_LR, betas=cfg.SOLVER.BETAS)
    else:
        raise ValueError(
            "Solver type {} is not supported!".format(solver_name))

    optimizer = maybe_add_gradient_clipping(cfg, optimizer)
    print('Optimizer: ', optimizer.__class__.__name__)
    return optimizer


def _create_residual_name(cfg):
    if cfg.SOLVER.RESIDUAL_LR_FACTOR is None:
        return None, None

    assert cfg.MODEL.BLOCK_TYPE in [
        'rcan_block', 'rcan_block_dw', 'rcan_block_all_dw']

    name_list = []
    prefix = 'module.model.body.'
    for i in range(cfg.MODEL.N_RESGROUPS):
        for j in range(cfg.cfg.MODEL.N_RESBLOCKS):
            name_list.append(prefix + str(i) + '.body.' + str(j))
    return name_list, cfg.SOLVER.RESIDUAL_LR_FACTOR


"""
# 2. Build LR Scheduler
"""


def build_lr_scheduler(
        cfg: CfgNode,
        optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Build a LR scheduler from config.
    :param cfg:
    :param optimizer:
    :return torch.optim.lr_scheduler._LRScheduler:
    """
    name = cfg.SOLVER.LR_SCHEDULER_NAME
    if name == "WarmupMultiStepLR":
        return WarmupMultiStepLR(
            optimizer,
            cfg.SOLVER.STEPS,
            cfg.SOLVER.GAMMA,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
        )
    elif name == "WarmupCosineLR":
        return WarmupCosineLR(
            optimizer,
            cfg.SOLVER.ITERATION_TOTAL,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
            min_lr=cfg.SOLVER.MIN_LR,
        )
    elif name == "MultiStepLR":
        return MultiStepLR(
            optimizer,
            milestones=cfg.SOLVER.STEPS,
            gamma=cfg.SOLVER.GAMMA
        )
    elif name == "ReduceLROnPlateau":
        return ReduceLROnPlateau(
            optimizer,
            mode='min', factor=cfg.SOLVER.GAMMA, patience=1000,
            threshold=0.001, threshold_mode='rel', cooldown=0,
            min_lr=1e-06, eps=1e-08
        )
    else:
        raise ValueError("Unknown LR scheduler: {}".format(name))


"""
# 3. Build SWA model
"""


def build_swa_model(cfg: CfgNode,
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    is_pretrained: bool = False):
    # Instead of copying weights during initialization, the SWA model copys
    # the model weights when self.update_parameters is first called.
    # https://github.com/pytorch/pytorch/blob/1.7/torch/optim/swa_utils.py#L107

    # The SWA model needs to be constructed for all processes in distributed
    # training, otherwise the training can get stuck.
    swa_model = AveragedModel(model).to(device)
    if is_pretrained:
        swa_model.update_parameters(model)

    lr = cfg.SOLVER.BASE_LR
    lr *= cfg.SOLVER.SWA.LR_FACTOR
    swa_scheduler = SWALR(optimizer, swa_lr=lr)
    return swa_model, swa_scheduler
