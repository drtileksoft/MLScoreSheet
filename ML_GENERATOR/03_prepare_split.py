#!/usr/bin/env python3
# připraví train/val split pro:
#  - YOLO detekci fiducialů (dataset/detect/{images,labels})
#  - klasifikaci checkboxů (dataset/cls/{filled,empty})
#
# Použití:
#   python prepare_split.py --val-ratio 0.1 --seed 42 --clean --debug
#
# Struktura vstupu (z tvého generátoru):
#   dataset/
#     detect/images/*.jpg
#     detect/labels/*.txt
#     cls/filled/*.png
#     cls/empty/*.png
#
# Výstup:
#   datasets/
#     yolo/images/{train,val}/*.jpg
#     yolo/labels/{train,val}/*.txt
#     yolo/data.yaml
#     cls/{train,val}/{filled,empty}/*.png

import argparse
import random
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

# ------------------------------
# Log util
# ------------------------------
def log(msg: str): print(msg, flush=True)
def dbg(msg: str, enabled: bool): 
    if enabled: print(f"[DEBUG] {msg}", flush=True)
def warn(msg: str): print(f"[WARN ] {msg}", flush=True)
def err(msg: str): print(f"[ERROR] {msg}", flush=True)

# ------------------------------
# Config
# ------------------------------
SRC = Path("dataset")
SRC_DET_IMG = SRC/"detect"/"images"
SRC_DET_LBL = SRC/"detect"/"labels"
SRC_CLS     = SRC/"cls"

OUT = Path("datasets")
YOLO_IMG_TR = OUT/"yolo"/"images"/"train"
YOLO_IMG_VA = OUT/"yolo"/"images"/"val"
YOLO_LBL_TR = OUT/"yolo"/"labels"/"train"
YOLO_LBL_VA = OUT/"yolo"/"labels"/"val"
YOLO_YAML   = OUT/"yolo"/"data.yaml"

CLS_TR = OUT/"cls"/"train"
CLS_VA = OUT/"cls"/"val"

# ------------------------------
# Helpers
# ------------------------------
def ensure_dirs(debug=False):
    for p in [YOLO_IMG_TR, YOLO_IMG_VA, YOLO_LBL_TR, YOLO_LBL_VA, CLS_TR, CLS_VA]:
        p.mkdir(parents=True, exist_ok=True)
        dbg(f"Ensured dir: {p}", debug)

def clean_dirs(debug=False):
    if OUT.exists():
        dbg(f"Removing: {OUT}", debug)
        shutil.rmtree(OUT)

def list_images(folder: Path, exts=(".jpg",".jpeg",".png",".bmp")) -> List[Path]:
    files = []
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
    n_val = max(1, int(len(items)*val_ratio)) if len(items) else 0
    return items[n_val:], items[:n_val]  # train, val

def write_yaml(path: Path, body: str, debug=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    dbg(f"Wrote YAML: {path}", debug)

# ------------------------------
# Core
# ------------------------------
def split_yolo(val_ratio: float, seed: int, limit: int|None, dry: bool, debug: bool):
    if not SRC_DET_IMG.exists(): 
        err(f"Missing folder: {SRC_DET_IMG}"); return 0,0,0
    if not SRC_DET_LBL.exists(): 
        err(f"Missing folder: {SRC_DET_LBL}"); return 0,0,0

    imgs = list_images(SRC_DET_IMG, (".jpg",".jpeg",".png"))
    dbg(f"Found {len(imgs)} detection images in {SRC_DET_IMG}", debug)
    if debug and imgs[:5]:
        dbg("Sample images: " + ", ".join(p.name for p in imgs[:5]), debug)

    # kontrola labelů
    missing = []
    for p in imgs:
        lbl = SRC_DET_LBL/(p.stem + ".txt")
        if not lbl.exists():
            missing.append(p.name)
    if missing:
        warn(f"Missing {len(missing)} label(s). "
             f"First few: {missing[:5]} ... (files will be skipped)")
        imgs = [p for p in imgs if (SRC_DET_LBL/(p.stem + ".txt")).exists()]
        dbg(f"After removing missing-label images: {len(imgs)} remain", debug)

    tr, va = split_list(imgs, val_ratio, seed, limit, debug)
    log(f"YOLO: total={len(imgs)}, train={len(tr)}, val={len(va)}")

    # copy
    for p in tr:
        copy2(p, YOLO_IMG_TR/p.name, dry, debug)
        copy2(SRC_DET_LBL/(p.stem + ".txt"), YOLO_LBL_TR/(p.stem + ".txt"), dry, debug)
    for p in va:
        copy2(p, YOLO_IMG_VA/p.name, dry, debug)
        copy2(SRC_DET_LBL/(p.stem + ".txt"), YOLO_LBL_VA/(p.stem + ".txt"), dry, debug)

    # YAML
    yaml_body = (
        "path: ./datasets/yolo\n"
        "train: images/train\n"
        "val: images/val\n"
        "names: [TL, TR, BR, BL]\n"
    )
    if not YOLO_YAML.exists() or debug:
        write_yaml(YOLO_YAML, yaml_body, debug)

    return len(imgs), len(tr), len(va)

def split_cls(val_ratio: float, seed: int, limit: int|None, dry: bool, debug: bool):
    base = SRC_CLS
    filled = list_images(base/"filled", (".png",".jpg",".jpeg"))
    empty  = list_images(base/"empty",  (".png",".jpg",".jpeg"))
    dbg(f"Found cls filled={len(filled)}, empty={len(empty)}", debug)
    if debug and (filled[:3] or empty[:3]):
        dbg("Sample filled: " + ", ".join(p.name for p in filled[:3]), debug)
        dbg("Sample empty : " + ", ".join(p.name for p in empty[:3]), debug)

    # balanced limit (when --limit is used)
    def split_and_copy(files: List[Path], cls_name: str):
        tr, va = split_list(files, val_ratio, seed, limit, debug)
        for i,f in enumerate(tr): copy2(f, (CLS_TR/cls_name)/f.name, dry, debug)
        for i,f in enumerate(va): copy2(f, (CLS_VA/cls_name)/f.name, dry, debug)
        return len(files), len(tr), len(va)

    filled_all, filled_tr, filled_va = split_and_copy(filled, "filled")
    empty_all , empty_tr , empty_va  = split_and_copy(empty , "empty")

    log(f"CLS: total={filled_all+empty_all} "
        f"(filled {filled_all}, empty {empty_all})  "
        f"train={filled_tr+empty_tr}, val={filled_va+empty_va}")
    return (filled_all+empty_all), (filled_tr+empty_tr), (filled_va+empty_va)

# ------------------------------
# CLI
# ------------------------------
def main():
    ap = argparse.ArgumentParser(description="Prepare train/val split for YOLO+CLS with verbose debug.")
    ap.add_argument("--val-ratio", type=float, default=0.10, help="validation ratio (0..1)")
    ap.add_argument("--seed", type=int, default=42, help="random seed for split")
    ap.add_argument("--clean", action="store_true", help="remove ./datasets before writing")
    ap.add_argument("--dry-run", action="store_true", help="print actions only, copy nothing")
    ap.add_argument("--limit", type=int, default=None, help="use at most N images per task (quick test)")
    ap.add_argument("--debug", action="store_true", help="verbose debug output")
    args = ap.parse_args()

    log("=== prepare_split.py ===")
    log(f"SRC: {SRC.resolve()}")
    log(f"OUT: {OUT.resolve()}")
    log(f"val-ratio={args.val_ratio}  seed={args.seed}  clean={args.clean}  dry-run={args.dry_run}  limit={args.limit}  debug={args.debug}")

    if args.clean:
        clean_dirs(args.debug)

    ensure_dirs(args.debug)

    # YOLO split
    y_total, y_tr, y_va = split_yolo(args.val_ratio, args.seed, args.limit, args.dry_run, args.debug)
    # CLS split
    c_total, c_tr, c_va = split_cls(args.val_ratio, args.seed, args.limit, args.dry_run, args.debug)

    log("--- SUMMARY ---")
    log(f"YOLO: total={y_total}, train={y_tr}, val={y_va}")
    log(f"CLS : total={c_total}, train={c_tr}, val={c_va}")
    if args.dry_run:
        log("DRY-RUN mode: no files were copied.")
    else:
        log(f"Output ready in: {OUT.resolve()}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
