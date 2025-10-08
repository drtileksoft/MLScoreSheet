#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Připraví train/val split pouze pro YOLO detekci fiducialů.

Vstup (z generátoru):
  dataset/
    detect/images/*.jpg|*.jpeg|*.png
    detect/labels/*.txt     # YOLO formát: cls cx cy w h (normalized)

Výstup:
  datasets/
    yolo/images/{train,val}/*.jpg|*.png
    yolo/labels/{train,val}/*.txt
    yolo/data.yaml

Použití:
  python prepare_split_yolo.py --val-ratio 0.1 --seed 42 --clean --debug
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# ------------------------------
# Log util
# ------------------------------
def log(msg: str): print(msg, flush=True)
def dbg(msg: str, enabled: bool):
    if enabled: print(f"[DEBUG] {msg}", flush=True)
def warn(msg: str): print(f"[WARN ] {msg}", flush=True)
def err(msg: str): print(f"[ERROR] {msg}", flush=True)

# ------------------------------
# Cesty
# ------------------------------
SRC = Path("dataset")
SRC_DET_IMG = SRC / "detect" / "images"
SRC_DET_LBL = SRC / "detect" / "labels"

OUT = Path("datasets")
YOLO_IMG_TR = OUT / "yolo" / "images" / "train"
YOLO_IMG_VA = OUT / "yolo" / "images" / "val"
YOLO_LBL_TR = OUT / "yolo" / "labels" / "train"
YOLO_LBL_VA = OUT / "yolo" / "labels" / "val"
YOLO_YAML   = OUT / "yolo" / "data.yaml"

# ------------------------------
# Helpers
# ------------------------------
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

def ensure_dirs(debug=False):
    for p in [YOLO_IMG_TR, YOLO_IMG_VA, YOLO_LBL_TR, YOLO_LBL_VA]:
        p.mkdir(parents=True, exist_ok=True)
        dbg(f"Ensured dir: {p}", debug)

def clean_dirs(debug=False):
    if OUT.exists():
        dbg(f"Removing: {OUT}", debug)
        shutil.rmtree(OUT)

def list_images(folder: Path, exts=IMG_EXTS) -> List[Path]:
    files: List[Path] = []
    for e in exts:
        files.extend(folder.glob(f"*{e}"))
    return sorted(files)

def copy2(src: Path, dst: Path, dry: bool, debug=False):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dry:
        dbg(f"[DRY] copy {src} -> {dst}", debug)
    else:
        shutil.copy2(src, dst)
        dbg(f"copied {src.name} -> {dst}", debug)

def split_list(items: List[Path], val_ratio: float, seed: int, limit: int|None, debug=False) -> Tuple[List[Path], List[Path]]:
    if limit is not None:
        items = items[:limit]
        dbg(f"Applied --limit: using first {len(items)} items", debug)
    rng = random.Random(seed)
    items = items.copy()
    rng.shuffle(items)
    n_val = max(1, int(len(items) * val_ratio)) if len(items) and val_ratio > 0 else 0
    return items[n_val:], items[:n_val]  # train, val

def write_yaml(path: Path, body: str, debug=False, force=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if force or (not path.exists()):
        path.write_text(body, encoding="utf-8")
        dbg(f"Wrote YAML: {path}", debug)
    else:
        dbg(f"YAML exists, not overwritten: {path}", debug)

# ------------------------------
# Label validation
# ------------------------------
def validate_label_file(lbl_path: Path, n_classes: int, strict: bool, debug: bool) -> Tuple[bool, Dict[int,int]]:
    """
    Základní validace YOLO labelu.
    - každý řádek: class_id cx cy w h (>=5 čísel; ostatní hodnoty ignorujeme)
    - class_id v rozsahu 0..n_classes-1
    Vrací: (is_valid, per-class count)
    """
    cls_counts: Dict[int,int] = {}
    try:
        lines = lbl_path.read_text(encoding="utf-8").strip().splitlines()
    except Exception as e:
        warn(f"Nelze číst label {lbl_path.name}: {e}")
        return (not strict), cls_counts  # ve strict režimu neprojde

    ok = True
    for i, line in enumerate(lines, start=1):
        parts = line.strip().split()
        if len(parts) < 5:
            warn(f"{lbl_path.name}: řádek {i} má méně než 5 hodnot -> '{line}'")
            ok = False
            if strict: break
            else: continue
        # class id
        try:
            cid = int(float(parts[0]))
        except ValueError:
            warn(f"{lbl_path.name}: řádek {i} má nečíselný class id -> '{parts[0]}'")
            ok = False
            if strict: break
            else: continue
        if not (0 <= cid < n_classes):
            warn(f"{lbl_path.name}: class id {cid} mimo rozsah 0..{n_classes-1}")
            ok = False
            if strict: break
        cls_counts[cid] = cls_counts.get(cid, 0) + 1

        # bbox floats (nemusíme striktně ověřovat rozsah 0..1, jen čísla)
        for j in range(1, 5):
            try:
                float(parts[j])
            except ValueError:
                warn(f"{lbl_path.name}: řádek {i} hodnota '{parts[j]}' není číslo")
                ok = False
                if strict: break
        if strict and not ok:
            break

    return (ok or not strict), cls_counts

# ------------------------------
# Core
# ------------------------------
def split_yolo(val_ratio: float, seed: int, limit: int|None, dry: bool, debug: bool,
               names: List[str], strict: bool, force_yaml: bool):
    if not SRC_DET_IMG.exists():
        err(f"Missing folder: {SRC_DET_IMG}"); return 0,0,0, {}
    if not SRC_DET_LBL.exists():
        err(f"Missing folder: {SRC_DET_LBL}"); return 0,0,0, {}

    imgs = list_images(SRC_DET_IMG)
    dbg(f"Found {len(imgs)} detection images in {SRC_DET_IMG}", debug)
    if debug and imgs[:5]:
        dbg("Sample images: " + ", ".join(p.name for p in imgs[:5]), debug)

    # chybějící labely
    missing = []
    for p in imgs:
        if not (SRC_DET_LBL / f"{p.stem}.txt").exists():
            missing.append(p.name)
    if missing:
        warn(f"Missing {len(missing)} label(s). First few: {missing[:5]} ... (images will be skipped)")
        imgs = [p for p in imgs if (SRC_DET_LBL / f"{p.stem}.txt").exists()]
        dbg(f"After removing missing-label images: {len(imgs)} remain", debug)

    # orphaned labels (nepárované k žádnému obrázku)
    orphan = []
    for lbl in SRC_DET_LBL.glob("*.txt"):
        img_exists = any((SRC_DET_IMG / f"{lbl.stem}{ext}").exists() for ext in IMG_EXTS)
        if not img_exists:
            orphan.append(lbl.name)
    if orphan:
        warn(f"Found {len(orphan)} orphan label(s) without matching image. First few: {orphan[:5]}")

    # validace labelů + souhrn tříd
    n_classes = len(names)
    total_class_counts: Dict[int,int] = {}
    valid_imgs: List[Path] = []
    invalid_imgs: List[str] = []

    for p in imgs:
        lbl = SRC_DET_LBL / f"{p.stem}.txt"
        is_ok, counts = validate_label_file(lbl, n_classes, strict=strict, debug=debug)
        if is_ok:
            valid_imgs.append(p)
            for cid, cnt in counts.items():
                total_class_counts[cid] = total_class_counts.get(cid, 0) + cnt
        else:
            invalid_imgs.append(p.name)

    if invalid_imgs:
        warn(f"Invalid labels in {len(invalid_imgs)} file(s). First few: {invalid_imgs[:5]} "
             f"{'(skipped due to --strict)' if strict else '(kept: non-strict mode)'}")
        if strict:
            imgs = valid_imgs
        # non-strict: necháme obrázky, i když label má drobné problémy

    tr, va = split_list(imgs, val_ratio, seed, limit, debug)
    log(f"YOLO: total={len(imgs)}, train={len(tr)}, val={len(va)}")

    # kopírování
    for p in tr:
        copy2(p, YOLO_IMG_TR / p.name, dry, debug)
        copy2(SRC_DET_LBL / f"{p.stem}.txt", YOLO_LBL_TR / f"{p.stem}.txt", dry, debug)
    for p in va:
        copy2(p, YOLO_IMG_VA / p.name, dry, debug)
        copy2(SRC_DET_LBL / f"{p.stem}.txt", YOLO_LBL_VA / f"{p.stem}.txt", dry, debug)

    # YAML
    yaml_body = (
        "path: ./datasets/yolo\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {n_classes}\n"
        "names: [" + ", ".join(names) + "]\n"
    )
    write_yaml(YOLO_YAML, yaml_body, debug, force=force_yaml)

    return len(imgs), len(tr), len(va), total_class_counts

# ------------------------------
# CLI
# ------------------------------
def parse_names_arg(raw: str) -> List[str]:
    # umožní "TL,TR,BR,BL" nebo "0:TL,1:TR,2:BR,3:BL" – výstupem je jen seznam jmen
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    # podpora "i:name" i prostého "name"
    names: Dict[int,str] = {}
    next_idx = 0
    for p in parts:
        if ":" in p:
            i_s, name = p.split(":", 1)
            i = int(i_s.strip())
            names[i] = name.strip()
        else:
            names[next_idx] = p
            next_idx += 1
    if not names:
        raise ValueError("Argument --names nesmí být prázdný.")
    max_idx = max(names.keys())
    res = [names.get(i, f"class_{i}") for i in range(max_idx + 1)]
    return res

def main():
    ap = argparse.ArgumentParser(description="Prepare train/val split for YOLO fiducials (no CLS).")
    ap.add_argument("--val-ratio", type=float, default=0.10, help="validation ratio (0..1)")
    ap.add_argument("--seed", type=int, default=42, help="random seed for split")
    ap.add_argument("--clean", action="store_true", help="remove ./datasets before writing")
    ap.add_argument("--dry-run", action="store_true", help="print actions only, copy nothing")
    ap.add_argument("--limit", type=int, default=None, help="use at most N images (quick test)")
    ap.add_argument("--debug", action="store_true", help="verbose debug output")
    ap.add_argument("--strict", action="store_true", help="strict label validation (skip invalid)")
    ap.add_argument("--force-yaml", action="store_true", help="overwrite data.yaml if exists")
    ap.add_argument("--names", type=str, default="TL,TR,BR,BL",
                    help="class names as comma list; supports 'i:name' mapping, e.g. '0:TL,1:TR,2:BR,3:BL'")

    args = ap.parse_args()
    try:
        names = parse_names_arg(args.names)
    except Exception as e:
        err(f"Chybný --names: {e}")
        return 2

    log("=== prepare_split_yolo.py ===")
    log(f"SRC: {SRC.resolve()}")
    log(f"OUT: {OUT.resolve()}")
    log(f"val-ratio={args.val_ratio}  seed={args.seed}  clean={args.clean}  "
        f"dry-run={args.dry_run}  limit={args.limit}  debug={args.debug}  "
        f"strict={args.strict}  force-yaml={args.force_yaml}  names={names}")

    if args.clean:
        clean_dirs(args.debug)

    ensure_dirs(args.debug)

    total, n_tr, n_va, cls_counts = split_yolo(
        args.val_ratio, args.seed, args.limit, args.dry_run, args.debug,
        names=names, strict=args.strict, force_yaml=args.force_yaml
    )

    # Souhrn tříd
    if cls_counts:
        ordered = sorted(cls_counts.items(), key=lambda kv: kv[0])
        pretty = ", ".join(f"{names[cid]}:{cnt}" for cid, cnt in ordered if cid < len(names))
        log(f"Class distribution (by objects in labels): {pretty}")

    log("--- SUMMARY ---")
    log(f"YOLO: total={total}, train={n_tr}, val={n_va}")
    if args.dry_run:
        log("DRY-RUN mode: no files were copied.")
    else:
        log(f"Output ready in: {OUT.resolve()}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
