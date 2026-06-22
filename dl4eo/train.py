"""
dl4eo.train — model builder and Lightning training wrapper.

Supported models (universal / no dataset-specific weights)
-----------------------------------------------------------
smp   : unet | unet++ | deeplabv3+ | fpn | pspnet | linknet | pan | manet
        backbone = any timm/torchvision encoder  (e.g. resnet34, efficientnet-b3)

segformer-b{0..5}
        timm Mix-Transformer backbone + lightweight all-MLP decoder head
        backbone = "mit_b0" … "mit_b5"

vit-tiny | vit-small | vit-base
        timm ViT backbone + patch-shuffle pixel decoder (trained from scratch)
        backbone = "vit_tiny_patch16_224" | "vit_small_patch16_224" | …

Usage
-----
    import dl4eo

    # Auto split + train
    dl4eo.train(
        data_dir="/path/to/output",
        model="unet",
        backbone="resnet34",
        in_channels=10,      # None → auto-detect from first patch
        num_classes=2,
        weights=None,        # None (scratch) or "imagenet"
        split_strategy="temporal",
        batch_size=16,
        max_epochs=50,
    )

    # User-provided splits
    dl4eo.train(
        data_dir="/path/to/output",
        model="segformer",
        backbone="mit_b2",
        split_file="/path/to/output/splits.json",
    )
"""

from __future__ import annotations

import os
import json

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import lightning as L
    import torchmetrics
    import segmentation_models_pytorch as smp
    import timm
except ImportError as e:
    raise ImportError(
        f"Training dependencies missing ({e}).  "
        "Install with:  pip install dl4eo[train]"
    ) from e

from torch.utils.data import DataLoader
from dl4eo.io import PatchDataset
from dl4eo import splits as _splits
from dl4eo import stats as _stats


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

_SMP_MODELS = {
    "unet":       smp.Unet,
    "unet++":     smp.UnetPlusPlus,
    "deeplabv3+": smp.DeepLabV3Plus,
    "fpn":        smp.FPN,
    "pspnet":     smp.PSPNet,
    "linknet":    smp.Linknet,
    "pan":        smp.PAN,
    "manet":      smp.MAnet,
}

def _default_segformer_backbone() -> str:
    """Return mit_b0 if available (timm < 1.0), else fall back to swin_tiny."""
    try:
        import timm as _timm
        if any("mit_b0" in m for m in _timm.list_models("mit_b*")):
            return "mit_b0"
    except Exception:
        pass
    return "swin_tiny_patch4_window7_224"


_DEFAULT_BACKBONES = {
    **{k: "resnet34" for k in _SMP_MODELS},
    "segformer": _default_segformer_backbone(),
    "vit-tiny":  "vit_tiny_patch16_224",
    "vit-small": "vit_small_patch16_224",
    "vit-base":  "vit_base_patch16_224",
}


# ---------------------------------------------------------------------------
# SegFormer-style: any hierarchical timm backbone + lightweight all-MLP decoder
#
# Backbone options (pass as `backbone=` to build_model / dl4eo.train):
#   MiT (original SegFormer encoders — requires timm < 1.0):
#       mit_b0  mit_b1  mit_b2  mit_b3  mit_b4  mit_b5
#   Swin Transformer (default when MiT is not available in timm ≥ 1.0):
#       swin_tiny_patch4_window7_224   swin_small_patch4_window7_224
#       swin_base_patch4_window7_224   swin_large_patch4_window7_224
#   Any other timm hierarchical backbone with features_only=True support:
#       convnext_tiny  pvt_v2_b0  efficientformer_l1  ...
#
# Note: when MiT backbones are used this matches the original SegFormer
# architecture (Xie et al., 2021).  With Swin or other backbones the decoder
# head is identical but the encoder differs — users can pick whichever timm
# backbone is available in their environment.
# ---------------------------------------------------------------------------

class _SegFormerDecoder(nn.Module):
    def __init__(self, in_channels_list: list[int], embed_dim: int, num_classes: int):
        super().__init__()
        self.projs = nn.ModuleList([nn.Conv2d(c, embed_dim, 1) for c in in_channels_list])
        self.fuse  = nn.Sequential(
            nn.Conv2d(embed_dim * len(in_channels_list), embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(embed_dim, num_classes, 1)

    def forward(self, features: list, target_hw: tuple) -> torch.Tensor:
        h, w = target_hw
        upsampled = [
            F.interpolate(proj(feat), size=(h, w), mode="bilinear", align_corners=False)
            for feat, proj in zip(features, self.projs)
        ]
        return self.head(self.fuse(torch.cat(upsampled, dim=1)))


class _SegFormerModel(nn.Module):
    def __init__(self, backbone: str, in_channels: int, num_classes: int, img_size: int = 256):
        super().__init__()
        # Some backbones (Swin) require img_size at construction; others ignore it.
        try:
            self.encoder = timm.create_model(
                backbone, pretrained=False, in_chans=in_channels,
                features_only=True, img_size=img_size,
            )
        except TypeError:
            self.encoder = timm.create_model(
                backbone, pretrained=False, in_chans=in_channels, features_only=True
            )
        chs = self.encoder.feature_info.channels()
        self.decoder = _SegFormerDecoder(chs, embed_dim=256, num_classes=num_classes)
        # Backbones that return features in channels-last (B,H,W,C) format
        _ch_last = ("swin", "coat", "van", "twins", "efficientformer",
                    "metaformer", "fastvit", "pvt")
        self._channels_last_backbone = any(k in backbone for k in _ch_last)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        if self._channels_last_backbone:
            feats = [f.permute(0, 3, 1, 2).contiguous() for f in feats]
        return self.decoder(feats, x.shape[-2:])


# ---------------------------------------------------------------------------
# ViT: timm backbone + patch-pixel decoder (from scratch)
# ---------------------------------------------------------------------------

class _ViTSegModel(nn.Module):
    """ViT encoder + simple patch-shuffle pixel decoder."""

    def __init__(self, backbone: str, in_channels: int, num_classes: int, img_size: int):
        super().__init__()
        self.encoder = timm.create_model(
            backbone,
            pretrained=False,
            in_chans=in_channels,
            img_size=img_size,
            num_classes=0,
            global_pool="",
        )
        embed_dim  = self.encoder.embed_dim
        patch_size = self.encoder.patch_embed.patch_size
        if isinstance(patch_size, (tuple, list)):
            patch_size = patch_size[0]
        self.patch_size  = patch_size
        self.num_classes = num_classes
        self.img_size    = img_size
        self.n_side      = img_size // patch_size

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, patch_size * patch_size * num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        tokens = self.encoder.forward_features(x)   # [B, 1+N, D]
        tokens = tokens[:, 1:]                       # drop CLS  → [B, N, D]
        tokens = self.head(tokens)                   # [B, N, P*P*C]
        p, n, nc = self.patch_size, self.n_side, self.num_classes
        tokens = tokens.view(B, n, n, p, p, nc)
        tokens = tokens.permute(0, 5, 1, 3, 2, 4).contiguous()
        return tokens.view(B, nc, self.img_size, self.img_size)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(
    model: str,
    backbone: str = None,
    in_channels: int = 10,
    num_classes: int = 2,
    weights=None,
    img_size: int = 256,
) -> nn.Module:
    """
    Build a segmentation model.

    Parameters
    ----------
    model : str
        Model family.  One of the keys listed in dl4eo.train.SUPPORTED_MODELS.
    backbone : str, optional
        Encoder name.  Defaults per model are resolved at runtime:
        - smp models (unet, fpn, …) : "resnet34" — any timm/torchvision encoder works
        - segformer                  : "mit_b0" if available in your timm build,
                                       else "swin_tiny_patch4_window7_224".
                                       Pass mit_b0..b5 explicitly when timm < 1.0;
                                       use swin_*/convnext_* for timm ≥ 1.0.
        - vit-tiny / small / base    : corresponding vit_*_patch16_224 backbone
    in_channels : int
        Number of input channels (default 10).
    num_classes : int
        Output classes including background (default 2).
    weights : str or None
        None = random init; "imagenet" = ImageNet pretrained (smp only, 3-ch encoders).
    img_size : int
        Spatial size of input patches in pixels (used by ViT and Swin, default 256).

    Returns
    -------
    nn.Module
    """
    model = model.lower().strip()
    backbone = backbone or _DEFAULT_BACKBONES.get(model, "resnet34")

    if model in _SMP_MODELS:
        encoder_weights = weights if weights else None
        return _SMP_MODELS[model](
            encoder_name=backbone,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
        )

    if model == "segformer":
        return _SegFormerModel(backbone, in_channels, num_classes, img_size=img_size)

    if model.startswith("vit"):
        # Normalise: "vit-small" → "vit_small_patch16_224"
        if "_patch" not in backbone:
            variant_map = {
                "vit-tiny":  "vit_tiny_patch16_224",
                "vit-small": "vit_small_patch16_224",
                "vit-base":  "vit_base_patch16_224",
                "vit":       "vit_small_patch16_224",
            }
            backbone = variant_map.get(model, backbone)
        return _ViTSegModel(backbone, in_channels, num_classes, img_size)

    raise ValueError(
        f"Unknown model '{model}'.  "
        f"Supported: {list(_SMP_MODELS)} + ['segformer', 'vit-tiny', 'vit-small', 'vit-base']"
    )


SUPPORTED_MODELS = list(_SMP_MODELS) + ["segformer", "vit-tiny", "vit-small", "vit-base"]


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------

class SegmentationModule(L.LightningModule):
    """
    Generic Lightning module for binary / multi-class segmentation.

    Loss
    ----
    "dice_ce"  – DiceLoss + CrossEntropyLoss (default, works well for imbalanced masks)
    "dice"     – DiceLoss only
    "ce"       – CrossEntropyLoss only
    "focal"    – FocalLoss (from smp)
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int = 2,
        lr: float = 1e-3,
        loss: str = "dice_ce",
    ):
        super().__init__()
        self.model       = model
        self.num_classes = num_classes
        self.lr          = lr
        self.loss_name   = loss
        self.save_hyperparameters(ignore=["model"])

        mode = "multiclass" if num_classes > 2 else "binary"

        # Loss functions
        self._dice = smp.losses.DiceLoss(mode=mode, from_logits=True)
        self._ce   = nn.CrossEntropyLoss() if num_classes > 2 \
                     else nn.BCEWithLogitsLoss()
        self._focal = smp.losses.FocalLoss(mode=mode)

        # Metrics
        mk = {"task": "multiclass" if num_classes > 2 else "binary",
              "num_classes": num_classes}
        self.train_iou = torchmetrics.JaccardIndex(**mk)
        self.val_iou   = torchmetrics.JaccardIndex(**mk)
        self.val_f1    = torchmetrics.F1Score(**mk)

    # ------------------------------------------------------------------
    def _loss(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.num_classes == 2:
            # Binary losses expect single-channel logit [B,1,H,W]; use positive-class channel.
            logit_pos = (logits[:, 1:2] if logits.shape[1] == 2 else logits).contiguous()
            logit_1d  = logit_pos.squeeze(1)   # [B,H,W] for BCE
            mask_f    = mask.float()
            if self.loss_name == "dice_ce":
                return self._dice(logit_pos, mask_f) + self._ce(logit_1d, mask_f)
            if self.loss_name == "dice":
                return self._dice(logit_pos, mask_f)
            if self.loss_name == "focal":
                return self._focal(logit_pos, mask_f)
            return self._ce(logit_1d, mask_f)           # "ce"
        else:
            if self.loss_name == "dice_ce":
                return self._dice(logits, mask) + self._ce(logits, mask)
            if self.loss_name == "dice":
                return self._dice(logits, mask)
            if self.loss_name == "focal":
                return self._focal(logits, mask)
            return self._ce(logits, mask)                # "ce"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, _):
        logits = self(batch["image"])
        loss   = self._loss(logits, batch["mask"])
        preds  = logits.argmax(dim=1)   # [B, H, W] — works for both binary and multiclass
        self.train_iou(preds, batch["mask"])
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/iou",  self.train_iou, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, _):
        logits = self(batch["image"])
        loss   = self._loss(logits, batch["mask"])
        preds  = logits.argmax(dim=1)   # [B, H, W]
        self.val_iou(preds, batch["mask"])
        self.val_f1(preds,  batch["mask"])
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/iou",  self.val_iou, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val/f1",   self.val_f1,  prog_bar=True, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        opt   = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.max_epochs)
        return {"optimizer": opt, "lr_scheduler": sched}


# ---------------------------------------------------------------------------
# Lightning DataModule
# ---------------------------------------------------------------------------

class SegDataModule(L.LightningDataModule):
    """LightningDataModule wrapping PatchDataset for train / val / test."""

    def __init__(
        self,
        data_dir: str,
        split_file: str,
        stats_file: str = None,
        batch_size: int = 16,
        num_workers: int = 4,
        norm: str = "zscore",
        bands: list = None,
    ):
        super().__init__()
        self.data_dir    = data_dir
        self.split_file  = split_file
        self.stats_file  = stats_file
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.norm        = norm
        self.bands       = bands
        self._ds: dict[str, PatchDataset] = {}

    def setup(self, stage=None):
        kw = dict(
            data_dir   = self.data_dir,
            split_file = self.split_file,
            stats_file = self.stats_file,
            norm       = self.norm,
            bands      = self.bands,
        )
        for split in ("train", "val", "test"):
            try:
                self._ds[split] = PatchDataset(split=split, **kw)
            except ValueError:
                pass  # split may not exist (e.g. no test set)

    def _loader(self, split: str, shuffle: bool) -> DataLoader:
        return DataLoader(
            self._ds[split],
            batch_size  = self.batch_size,
            shuffle     = shuffle,
            num_workers = self.num_workers,
            pin_memory  = True,
            persistent_workers = self.num_workers > 0,
        )

    def train_dataloader(self):   return self._loader("train", shuffle=True)
    def val_dataloader(self):     return self._loader("val",   shuffle=False)
    def test_dataloader(self):    return self._loader("test",  shuffle=False)


# ---------------------------------------------------------------------------
# One-function training API
# ---------------------------------------------------------------------------

def train(
    data_dir: str,
    model: str = "unet",
    backbone: str = None,
    in_channels: int = None,
    num_classes: int = 2,
    weights=None,
    split_file: str = None,
    split_ratios: tuple = (0.7, 0.15, 0.15),
    split_strategy: str = "random",
    split_seed: int = 42,
    norm: str = "zscore",
    stats_file: str = None,
    bands: list = None,
    batch_size: int = 16,
    num_workers: int = 4,
    lr: float = 1e-3,
    loss: str = "dice_ce",
    max_epochs: int = 50,
    output_dir: str = None,
    accelerator: str = "auto",
    devices: int = 1,
    **trainer_kwargs,
) -> SegmentationModule:
    """
    Train a segmentation model on dl4eo patches.

    Parameters
    ----------
    data_dir : str
        Pipeline output directory.
    model : str
        Model name (see SUPPORTED_MODELS).
    backbone : str, optional
        Encoder backbone.  Defaults per model: resnet34 for smp, mit_b0 for segformer,
        vit_small_patch16_224 for vit-small.
    in_channels : int, optional
        Auto-detected from the first patch if None.
    num_classes : int
        Number of output classes including background (default 2).
    weights : str or None
        None (random init) or "imagenet" (smp only).
    split_file : str, optional
        Path to splits.json.  Auto-generated if None.
    split_ratios : tuple
        (train, val, test) fractions (default 0.7/0.15/0.15).
    split_strategy : str
        "random" | "temporal" | "spatial".
    split_seed : int
        RNG seed for split generation.
    norm : str or None
        "zscore" | "minmax" | "percentile" | None.
    stats_file : str, optional
        Path to stats.json.  Auto-computed from training split if None.
    bands : list of int, optional
        0-indexed band subset.  None = all bands.
    batch_size : int
        Mini-batch size (default 16).
    num_workers : int
        DataLoader workers (default 4).
    lr : float
        Initial learning rate (default 1e-3).
    loss : str
        "dice_ce" | "dice" | "ce" | "focal".
    max_epochs : int
        Training epochs (default 50).
    output_dir : str, optional
        Directory for checkpoints and logs.  Defaults to {data_dir}/runs/{model}.
    accelerator : str
        Lightning accelerator: "auto", "cpu", "gpu", "mps" (default "auto").
    devices : int
        Number of devices (default 1).
    **trainer_kwargs
        Extra kwargs forwarded to lightning.Trainer.

    Returns
    -------
    SegmentationModule  (trained, loaded from best checkpoint)
    """
    output_dir = output_dir or os.path.join(data_dir, "runs", f"{model}-{backbone or 'default'}")
    os.makedirs(output_dir, exist_ok=True)

    # --- Auto-detect in_channels ---
    if in_channels is None:
        import rasterio
        from dl4eo.io import _best_img_dir
        img_dir = _best_img_dir(data_dir)
        first   = sorted(f for f in os.listdir(img_dir) if f.endswith(".tif"))[0]
        with rasterio.open(os.path.join(img_dir, first)) as src:
            in_channels = src.count
        if bands:
            in_channels = len(bands)
        print(f"[INFO] Auto-detected in_channels={in_channels}")

    # --- Auto-detect img_size ---
    img_size = 256  # default; read from first patch
    try:
        import rasterio
        from dl4eo.io import _best_img_dir
        img_dir = _best_img_dir(data_dir)
        first   = sorted(f for f in os.listdir(img_dir) if f.endswith(".tif"))[0]
        with rasterio.open(os.path.join(img_dir, first)) as src:
            img_size = src.width
    except Exception:
        pass

    # --- Splits ---
    if split_file is None:
        split_file = os.path.join(data_dir, "splits.json")
        if not os.path.exists(split_file):
            print("[INFO] No splits.json found — generating splits")
            _splits.make_splits(
                data_dir,
                ratios   = split_ratios,
                strategy = split_strategy,
                seed     = split_seed,
            )

    # --- Stats ---
    if norm is not None:
        if stats_file is None:
            stats_file = os.path.join(data_dir, "stats.json")
        if not os.path.exists(stats_file):
            print("[INFO] No stats.json found — computing statistics from training split")
            _stats.compute(data_dir, split="train", split_file=split_file)

    # --- Build model ---
    print(f"[INFO] Building model: {model}  backbone: {backbone or _DEFAULT_BACKBONES.get(model, 'resnet34')}")
    net = build_model(
        model       = model,
        backbone    = backbone,
        in_channels = in_channels,
        num_classes = num_classes,
        weights     = weights,
        img_size    = img_size,
    )
    module = SegmentationModule(net, num_classes=num_classes, lr=lr, loss=loss)

    # --- DataModule ---
    dm = SegDataModule(
        data_dir    = data_dir,
        split_file  = split_file,
        stats_file  = stats_file,
        batch_size  = batch_size,
        num_workers = num_workers,
        norm        = norm,
        bands       = bands,
    )

    # --- Callbacks ---
    ckpt_cb = L.pytorch.callbacks.ModelCheckpoint(
        dirpath   = output_dir,
        filename  = "best-{epoch:02d}-{val/iou:.3f}",
        monitor   = "val/iou",
        mode      = "max",
        save_top_k = 1,
    )
    lr_cb = L.pytorch.callbacks.LearningRateMonitor(logging_interval="epoch")

    # --- Trainer ---
    trainer = L.Trainer(
        max_epochs  = max_epochs,
        accelerator = accelerator,
        devices     = devices,
        callbacks   = [ckpt_cb, lr_cb],
        default_root_dir = output_dir,
        log_every_n_steps = 1,
        **trainer_kwargs,
    )

    trainer.fit(module, dm)

    # Load best weights
    best_path = ckpt_cb.best_model_path
    if best_path and os.path.exists(best_path):
        module = SegmentationModule.load_from_checkpoint(best_path, model=net)
        print(f"[✓] Best checkpoint: {best_path}")

    return module
