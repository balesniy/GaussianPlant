#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA


FEATURE_EXTS = (".pt", ".pth", ".npy", ".npz")
MASK_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def list_feature_files(feature_dir):
    files = []
    for path in sorted(Path(feature_dir).iterdir()):
        if path.is_file() and path.suffix.lower() in FEATURE_EXTS:
            files.append(path)
    return files


def candidate_names(path):
    stem = path.stem
    name = path.name
    if path.suffix.lower() in FEATURE_EXTS:
        name = stem
    return [name, stem]


def find_matching_file(root, names, exts):
    root = Path(root)
    for name in names:
        candidates = [root / name]
        candidates.extend(root / f"{Path(name).stem}{ext}" for ext in exts)
        candidates.extend(root / f"{name}{ext}" for ext in exts)
        for candidate in candidates:
            if candidate.exists():
                return candidate
    return None


def load_feature(path, layout="auto"):
    suffix = path.suffix.lower()
    if suffix in (".pt", ".pth"):
        data = torch.load(path, map_location="cpu")
        if isinstance(data, dict):
            for key in ("features", "feature", "feature_map", "feat", "x"):
                if key in data:
                    data = data[key]
                    break
        feature = torch.as_tensor(data).detach().cpu().float().numpy()
    elif suffix == ".npz":
        data = np.load(path)
        key = "features" if "features" in data else data.files[0]
        feature = data[key].astype(np.float32)
    else:
        feature = np.load(path).astype(np.float32)

    if feature.ndim == 4 and feature.shape[0] == 1:
        feature = feature[0]
    if feature.ndim != 3:
        raise ValueError(f"Expected 3D feature map at {path}, got shape {feature.shape}")
    return to_hwc(feature, layout)


def to_hwc(feature, layout="auto"):
    if layout == "hwc":
        return feature
    if layout == "chw":
        return np.transpose(feature, (1, 2, 0))
    if feature.shape[0] > feature.shape[1] and feature.shape[0] > feature.shape[2]:
        return np.transpose(feature, (1, 2, 0))
    if feature.shape[2] > feature.shape[0] and feature.shape[2] > feature.shape[1]:
        return feature
    if feature.shape[-1] <= 4:
        return feature
    return np.transpose(feature, (1, 2, 0))


def load_mask(mask_path, size_hw, threshold):
    mask = Image.open(mask_path).convert("L")
    height, width = size_hw
    if mask.size != (width, height):
        mask = mask.resize((width, height), Image.BILINEAR)
    mask = np.asarray(mask, dtype=np.float32) / 255.0
    return mask > threshold


def sample_object_pixels(feature, mask, max_samples, rng):
    pixels = feature[mask]
    if pixels.shape[0] == 0:
        return pixels
    if pixels.shape[0] <= max_samples:
        return pixels
    idx = rng.choice(pixels.shape[0], size=max_samples, replace=False)
    return pixels[idx]


def transform_feature(feature, pca, out_dim):
    flat = feature.reshape(-1, feature.shape[-1])
    transformed = pca.transform(flat)[:, :out_dim]
    return transformed.reshape(feature.shape[0], feature.shape[1], out_dim).astype(np.float32)


def save_feature(path, feature_hwc, fmt):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "npy":
        np.save(path.with_suffix(".npy"), feature_hwc.astype(np.float32))
    else:
        torch.save(torch.from_numpy(feature_hwc.astype(np.float32)).permute(2, 0, 1).contiguous(), path.with_suffix(".pt"))


def main():
    parser = argparse.ArgumentParser(description="Fit PCA on object-mask DINO pixels and export normalized PCA feature maps.")
    parser.add_argument("--feature_dir", required=True, help="Directory with raw DINO feature maps (.pt/.pth/.npy/.npz).")
    parser.add_argument("--mask_dir", required=True, help="Directory with object masks named like the source images/features.")
    parser.add_argument("--output_dir", required=True, help="Directory for PCA-compressed feature maps.")
    parser.add_argument("--feature_layout", choices=["auto", "chw", "hwc"], default="auto")
    parser.add_argument("--dim", type=int, default=64, help="Output PCA dimension.")
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    parser.add_argument("--max_samples_per_view", type=int, default=4096)
    parser.add_argument("--max_total_samples", type=int, default=300000)
    parser.add_argument("--normalization_percentile", type=float, default=1.0, help="Use p and 100-p percentiles on object pixels. Set 0 for min/max.")
    parser.add_argument("--background", choices=["zero", "neutral"], default="neutral")
    parser.add_argument("--save_format", choices=["pt", "npy"], default="pt")
    parser.add_argument("--whiten", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    feature_files = list_feature_files(args.feature_dir)
    if not feature_files:
        raise ValueError(f"No feature files found in {args.feature_dir}")

    samples = []
    matched = []
    per_view_budget = max(args.max_samples_per_view, 1)
    for feature_path in feature_files:
        feature = load_feature(feature_path, args.feature_layout)
        mask_path = find_matching_file(args.mask_dir, candidate_names(feature_path), MASK_EXTS)
        if mask_path is None:
            print(f"[WARN] no mask for {feature_path.name}; skipping PCA samples")
            continue
        mask = load_mask(mask_path, feature.shape[:2], args.mask_threshold)
        view_samples = sample_object_pixels(feature, mask, per_view_budget, rng)
        if view_samples.shape[0] == 0:
            print(f"[WARN] empty object mask for {feature_path.name}; skipping PCA samples")
            continue
        samples.append(view_samples)
        matched.append((feature_path, mask_path))

    if not samples:
        raise ValueError("No object pixels were collected for PCA.")
    samples = np.concatenate(samples, axis=0).astype(np.float32)
    if samples.shape[0] > args.max_total_samples:
        idx = rng.choice(samples.shape[0], size=args.max_total_samples, replace=False)
        samples = samples[idx]

    out_dim = min(args.dim, samples.shape[1])
    print(f"[object-pca] fitting PCA dim={out_dim} on {samples.shape[0]} object pixels from {len(matched)} views")
    pca = PCA(n_components=out_dim, whiten=args.whiten, svd_solver="randomized", random_state=args.seed)
    pca.fit(samples)

    object_transformed_samples = pca.transform(samples)[:, :out_dim]
    if args.normalization_percentile > 0:
        low = np.percentile(object_transformed_samples, args.normalization_percentile, axis=0)
        high = np.percentile(object_transformed_samples, 100.0 - args.normalization_percentile, axis=0)
    else:
        low = object_transformed_samples.min(axis=0)
        high = object_transformed_samples.max(axis=0)
    scale = np.maximum(high - low, 1e-6).astype(np.float32)
    low = low.astype(np.float32)

    bg_value = 0.0 if args.background == "zero" else 0.5
    written = 0
    for feature_path in feature_files:
        feature = load_feature(feature_path, args.feature_layout)
        mask_path = find_matching_file(args.mask_dir, candidate_names(feature_path), MASK_EXTS)
        if mask_path is None:
            print(f"[WARN] no mask for {feature_path.name}; skipping output")
            continue
        mask = load_mask(mask_path, feature.shape[:2], args.mask_threshold)
        transformed = transform_feature(feature, pca, out_dim)
        transformed = np.clip((transformed - low) / scale, 0.0, 1.0)
        transformed[~mask] = bg_value
        save_feature(Path(args.output_dir) / feature_path.stem, transformed, args.save_format)
        written += 1

    metadata = {
        "dim": out_dim,
        "feature_dir": str(args.feature_dir),
        "mask_dir": str(args.mask_dir),
        "feature_layout": args.feature_layout,
        "mask_threshold": args.mask_threshold,
        "normalization_percentile": args.normalization_percentile,
        "background": args.background,
        "whiten": args.whiten,
        "num_views": written,
        "num_pca_samples": int(samples.shape[0]),
        "explained_variance_ratio": pca.explained_variance_ratio_.astype(float).tolist(),
        "normalization_low": low.astype(float).tolist(),
        "normalization_scale": scale.astype(float).tolist(),
    }
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output_dir) / "object_only_pca_meta.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[object-pca] wrote {written} maps to {args.output_dir}")


if __name__ == "__main__":
    main()
