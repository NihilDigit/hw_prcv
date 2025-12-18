"""PyTorch 2.9 训练工具：充分利用新特性加速"""

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from pathlib import Path
from typing import Callable
import json
import time


def setup_torch_optimizations():
    """设置 PyTorch 2.9 全局优化"""
    # TF32 加速矩阵运算 (RTX 30/40 系列)
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # cuDNN 优化
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    print(f"PyTorch {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"TF32: enabled")


def compile_model(model: nn.Module, mode: str = "reduce-overhead") -> nn.Module:
    """
    使用 torch.compile 编译模型加速

    Args:
        model: PyTorch 模型
        mode: 编译模式
            - "default": 平衡编译时间和性能
            - "reduce-overhead": 减少 Python 开销，适合小 batch
            - "max-autotune": 最大性能，编译时间长

    Returns:
        编译后的模型
    """
    return torch.compile(model, mode=mode)


class AMPTrainer:
    """自动混合精度训练器"""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        use_compile: bool = True,
        gradient_checkpointing: bool = False,
    ):
        self.device = device
        self.model = model.to(device)

        # torch.compile 加速
        if use_compile:
            self.model = compile_model(self.model)

        # 梯度检查点节省显存
        if gradient_checkpointing and hasattr(self.model, 'set_grad_checkpointing'):
            self.model.set_grad_checkpointing(True)

        self.optimizer = optimizer
        self.criterion = criterion

        # AMP
        self.scaler = GradScaler('cuda')

    def train_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        """单步训练"""
        self.model.train()
        inputs = inputs.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零

        with autocast('cuda'):
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss.item()

    @torch.no_grad()
    def eval_step(self, inputs: torch.Tensor) -> torch.Tensor:
        """推理"""
        self.model.eval()
        inputs = inputs.to(self.device, non_blocking=True)

        with autocast('cuda'):
            outputs = self.model(inputs)

        return outputs


class ExperimentLogger:
    """实验日志记录器"""

    def __init__(self, exp_dir: Path):
        self.exp_dir = Path(exp_dir)
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.exp_dir / "training_log.json"
        self.history = {"train_loss": [], "val_loss": [], "val_metric": [], "lr": [], "epoch_time": []}

    def log(self, epoch: int, train_loss: float, val_loss: float = None,
            val_metric: float = None, lr: float = None, epoch_time: float = None):
        """记录一个 epoch"""
        self.history["train_loss"].append(train_loss)
        if val_loss is not None:
            self.history["val_loss"].append(val_loss)
        if val_metric is not None:
            self.history["val_metric"].append(val_metric)
        if lr is not None:
            self.history["lr"].append(lr)
        if epoch_time is not None:
            self.history["epoch_time"].append(epoch_time)

        # 保存到文件
        with open(self.log_file, 'w') as f:
            json.dump(self.history, f, indent=2)

    def get_history(self) -> dict:
        return self.history


def poly_lr_scheduler(optimizer: torch.optim.Optimizer, epoch: int,
                      max_epochs: int, base_lr: float, power: float = 0.9):
    """Poly 学习率衰减策略"""
    lr = base_lr * (1 - epoch / max_epochs) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                    epoch: int, metric: float, path: Path, is_best: bool = False):
    """保存检查点"""
    # 处理 compiled model
    model_state = model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict()

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'metric': metric,
    }

    torch.save(checkpoint, path / "last.pth")
    if is_best:
        torch.save(checkpoint, path / "best.pth")


# 初始化时自动设置优化
setup_torch_optimizations()
