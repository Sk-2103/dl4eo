"""
dl4eo.eval — segmentation evaluation and GeoTIFF prediction export.

Usage
-----
    import dl4eo

    module = dl4eo.train(data_dir="/data/glacial_lakes", model="unet", ...)

    report = dl4eo.eval.evaluate(
        module,
        data_dir    = "/data/glacial_lakes",
        splits      = ("val", "test"),
        class_names = ["background", "lake"],
        output_dir  = "/data/glacial_lakes/eval",
        save_predictions = True,
    )
    # → prints formatted metric tables
    # → saves eval/predictions/val/*.tif  and  eval/predictions/test/*.tif
    # → saves eval/eval_report.json  and  eval/eval_report.txt
"""

import os
import json
import datetime
from pathlib import Path

import numpy as np
import torch
import rasterio

from .io import PatchDataset


# ─────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────

def _find_img_dir(data_dir: str) -> str:
    for name in ("stacked_with_sar", "stacked", "images"):
        d = os.path.join(data_dir, name)
        if os.path.isdir(d):
            return d
    raise FileNotFoundError(
        f"No image folder (stacked_with_sar / stacked / images) found under {data_dir}"
    )


def _confusion_matrix(preds: np.ndarray, targets: np.ndarray, num_classes: int) -> np.ndarray:
    """Accumulate pixel-level confusion matrix. cm[true, pred]."""
    cm   = np.zeros((num_classes, num_classes), dtype=np.int64)
    valid = (targets >= 0) & (targets < num_classes) & (preds >= 0) & (preds < num_classes)
    np.add.at(cm, (targets[valid], preds[valid]), 1)
    return cm


def _metrics_from_cm(cm: np.ndarray, num_classes: int):
    """
    Derive per-class IoU, F1, Precision, Recall and overall Accuracy + Kappa.
    Returns (per_class: dict[int → dict], mean: dict).
    """
    total     = int(cm.sum())
    per_class = {}
    ious, f1s, precs, recs = [], [], [], []

    for i in range(num_classes):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum()) - tp   # predicted i, not actually i
        fn = int(cm[i, :].sum()) - tp   # actually i, not predicted i

        iou  = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        prec = tp / (tp + fp)       if (tp + fp)      > 0 else 0.0
        rec  = tp / (tp + fn)       if (tp + fn)      > 0 else 0.0
        f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        per_class[i] = dict(iou=round(iou, 4), f1=round(f1, 4),
                            precision=round(prec, 4), recall=round(rec, 4))
        ious.append(iou);  f1s.append(f1)
        precs.append(prec); recs.append(rec)

    oa  = cm.diagonal().sum() / total if total > 0 else 0.0
    p_e = sum(cm[i, :].sum() * cm[:, i].sum() for i in range(num_classes)) / total**2 \
          if total > 0 else 0.0
    kappa = (oa - p_e) / (1 - p_e) if (1 - p_e) > 1e-9 else 0.0

    mean = dict(
        iou       = round(float(np.mean(ious)),  4),
        f1        = round(float(np.mean(f1s)),   4),
        precision = round(float(np.mean(precs)), 4),
        recall    = round(float(np.mean(recs)),  4),
        accuracy  = round(float(oa),             4),
        kappa     = round(float(kappa),          4),
    )
    return per_class, mean


def _print_table(split: str, n_patches: int, per_class: dict,
                 mean: dict, class_names: list, pred_dir=None):
    """Print a formatted segmentation metric table to the console."""
    W = 72
    CW = [22, 8, 8, 11, 8, 9]   # column widths

    def _sep(l, m, r):
        parts = [("─" * w) for w in CW]
        return "  " + l + m.join(parts) + r

    def _row(vals):
        cells = [str(v).center(CW[i]) for i, v in enumerate(vals)]
        return "  │" + "│".join(cells) + "│"

    print(f"\n{'═' * W}")
    print(f"  Evaluation — {split.upper()} split  ({n_patches} patches)")
    print(f"{'═' * W}")
    print(_sep("┌", "┬", "┐"))
    print(_row(["Class", "IoU", "F1", "Precision", "Recall", "Accuracy"]))
    print(_sep("├", "┼", "┤"))

    nc = len(per_class)
    for i in range(nc):
        name = class_names[i] if class_names and i < len(class_names) else f"class_{i}"
        m    = per_class[i]
        print(_row([f"{name} ({i})", m["iou"], m["f1"], m["precision"], m["recall"], "—"]))

    print(_sep("├", "┼", "┤"))
    print(_row(["Mean (mIoU)", mean["iou"], mean["f1"],
                mean["precision"], mean["recall"], mean["accuracy"]]))
    print(_sep("└", "┴", "┘"))
    print(f"\n  Cohen's Kappa : {mean['kappa']:.4f}")
    if pred_dir:
        print(f"  Predictions   → {pred_dir}")


def _save_prediction_geotiff(pred: np.ndarray, src_path: str, out_path: str):
    """Write prediction as single-band uint8 GeoTIFF, preserving CRS and transform."""
    with rasterio.open(src_path) as src:
        profile = src.profile.copy()

    profile.update(
        count    = 1,
        dtype    = "uint8",
        compress = "lzw",
        nodata   = 255,
    )
    # remove multi-band driver quirks
    for key in ("BIGTIFF",):
        profile.pop(key, None)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(pred.astype(np.uint8), 1)


def _save_text_report(report: dict, path: str, class_names: list):
    """Write a plain-text version of the report."""
    lines = []
    W = 72
    lines += [
        "=" * W,
        "  dl4eo Segmentation Evaluation Report",
        f"  Generated : {report.get('timestamp', '')}",
        f"  Data dir  : {report.get('data_dir', '')}",
        f"  Classes   : {', '.join(class_names)}",
        "=" * W,
    ]

    for split in ("val", "test"):
        if split not in report:
            continue
        r = report[split]
        lines += [
            f"\n{'─' * W}",
            f"  Split : {split.upper()}   ({r['n_patches']} patches)",
            f"{'─' * W}",
            f"  {'Class':<22} {'IoU':>8} {'F1':>8} {'Precision':>11} {'Recall':>8} {'Accuracy':>9}",
            f"  {'-'*22} {'-'*8} {'-'*8} {'-'*11} {'-'*8} {'-'*9}",
        ]
        for cname, m in r["per_class"].items():
            lines.append(
                f"  {cname:<22} {m['iou']:>8.4f} {m['f1']:>8.4f} "
                f"{m['precision']:>11.4f} {m['recall']:>8.4f} {'—':>9}"
            )
        lines.append(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*11} {'-'*8} {'-'*9}")
        m = r["mean"]
        lines.append(
            f"  {'Mean (mIoU)':<22} {m['iou']:>8.4f} {m['f1']:>8.4f} "
            f"{m['precision']:>11.4f} {m['recall']:>8.4f} {m['accuracy']:>9.4f}"
        )
        lines.append(f"\n  Cohen's Kappa : {m['kappa']:.4f}")

    lines += ["", "=" * W]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def load_module(ckpt_path: str, model: str, backbone: str = None,
                in_channels: int = None, num_classes: int = 2):
    """
    Reload a SegmentationModule from a saved checkpoint.

    Because the network architecture is not stored inside the checkpoint
    (only hyperparameters like lr, loss, num_classes are), you must supply
    the same model name and backbone used during training.

    Parameters
    ----------
    ckpt_path   : str   Path to the .ckpt file saved by dl4eo.train().
    model       : str   Model name used during training (e.g. "unet").
    backbone    : str   Backbone name (e.g. "resnet34"). None = default.
    in_channels : int   Number of input channels. Auto-detected from ckpt if None.
    num_classes : int   Number of output classes (default 2).

    Returns
    -------
    SegmentationModule  ready for inference or dl4eo.eval.evaluate().

    Example
    -------
        module = dl4eo.eval.load_module(
            "checkpoints/unet/best-epoch=10.ckpt",
            model="unet", backbone="resnet34", in_channels=10,
        )
    """
    import torch
    from .train import build_model, SegmentationModule

    # peek at hparams stored in the checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")
    hp   = ckpt.get("hyper_parameters", {})

    num_classes = hp.get("num_classes", num_classes)

    # infer in_channels from first conv weight if not given
    if in_channels is None:
        state = ckpt.get("state_dict", {})
        for k, v in state.items():
            if "weight" in k and v.ndim == 4:
                in_channels = v.shape[1]
                break
        if in_channels is None:
            raise ValueError(
                "Could not auto-detect in_channels from checkpoint. "
                "Please pass in_channels explicitly."
            )

    net    = build_model(model, in_channels=in_channels,
                         num_classes=num_classes, backbone=backbone)
    module = SegmentationModule.load_from_checkpoint(ckpt_path, model=net)
    return module


def evaluate(
    module,
    data_dir: str,
    split_file: str = None,
    stats_file: str = None,
    splits: tuple = ("val", "test"),
    output_dir: str = None,
    save_predictions: bool = True,
    num_classes: int = None,
    class_names: list = None,
    device: str = "auto",
):
    """
    Evaluate a trained SegmentationModule on val and/or test splits.

    Parameters
    ----------
    module : SegmentationModule
        Returned by dl4eo.train() or loaded from checkpoint.
    data_dir : str
        Base dataset directory (same as used for training).
    split_file : str, optional
        Path to splits.json. Auto-detected from data_dir if None.
    stats_file : str, optional
        Path to stats.json. Auto-detected from data_dir if None.
    splits : tuple
        Which splits to evaluate. Default ("val", "test").
    output_dir : str, optional
        Where to write predictions and report. Defaults to data_dir/eval/.
    save_predictions : bool
        If True, writes per-patch prediction GeoTIFFs in their original CRS.
    num_classes : int, optional
        Auto-inferred from module if None.
    class_names : list, optional
        Human-readable class labels e.g. ["background", "lake"].
    device : str
        "auto" | "cuda" | "cpu"

    Returns
    -------
    dict
        Full report keyed by split. Also saved as eval_report.json and
        eval_report.txt in output_dir.
    """
    # ── resolve paths ────────────────────────────────────────
    if output_dir is None:
        output_dir = os.path.join(data_dir, "eval")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if split_file is None:
        split_file = os.path.join(data_dir, "splits.json")
    if stats_file is None:
        stats_file = os.path.join(data_dir, "stats.json")

    split_file = split_file if os.path.exists(split_file) else None
    stats_file = stats_file if os.path.exists(stats_file) else None

    # ── resolve device ───────────────────────────────────────
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── resolve num_classes and class names ──────────────────
    if num_classes is None:
        num_classes = getattr(module, "num_classes", 2)
    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]

    # ── prepare model ────────────────────────────────────────
    module = module.to(device)
    module.eval()

    img_dir = _find_img_dir(data_dir)

    report = {
        "timestamp":   datetime.datetime.now().isoformat(),
        "data_dir":    data_dir,
        "num_classes": num_classes,
        "class_names": class_names,
    }

    # ── evaluate each split ──────────────────────────────────
    for split in splits:
        ds = PatchDataset(
            data_dir,
            split      = split,
            split_file = split_file,
            stats_file = stats_file,
        )

        if len(ds) == 0:
            print(f"  [SKIP] No patches for split='{split}'")
            continue

        pred_dir = os.path.join(output_dir, "predictions", split) if save_predictions else None
        if save_predictions:
            Path(pred_dir).mkdir(parents=True, exist_ok=True)

        cm = np.zeros((num_classes, num_classes), dtype=np.int64)

        print(f"\n  [INFO] Evaluating {split} — {len(ds)} patches ...", flush=True)

        for i in range(len(ds)):
            sample = ds[i]
            image  = sample["image"].unsqueeze(0).to(device)   # (1, C, H, W)
            mask   = sample["mask"].numpy()                     # (H, W)

            with torch.no_grad():
                logits = module.model(image)
                pred   = logits.argmax(dim=1).squeeze(0).cpu().numpy()  # (H, W)

            cm += _confusion_matrix(pred.ravel(), mask.ravel(), num_classes)

            if save_predictions:
                fname    = ds.files[i]
                src_path = os.path.join(img_dir, fname)
                out_path = os.path.join(pred_dir, fname)
                _save_prediction_geotiff(pred, src_path, out_path)

        per_class, mean = _metrics_from_cm(cm, num_classes)

        # name the per-class keys for the report
        per_class_named = {
            (class_names[i] if i < len(class_names) else f"class_{i}"): v
            for i, v in per_class.items()
        }

        _print_table(split, len(ds), per_class, mean, class_names, pred_dir=pred_dir)

        report[split] = {
            "n_patches":        len(ds),
            "per_class":        per_class_named,
            "mean":             mean,
            "confusion_matrix": cm.tolist(),
        }

    # ── save report ──────────────────────────────────────────
    json_path = os.path.join(output_dir, "eval_report.json")
    txt_path  = os.path.join(output_dir, "eval_report.txt")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    _save_text_report(report, txt_path, class_names)

    print(f"\n  Report (JSON) → {json_path}")
    print(f"  Report (text) → {txt_path}")

    return report
