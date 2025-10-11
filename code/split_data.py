# kfold_with_datautils.py
import shutil, os
from pathlib import Path
from sklearn.model_selection import KFold, StratifiedKFold, ShuffleSplit
import argparse

import yaml
from ultralytics import YOLO
from ultralytics.data.utils import img2label_paths  # helpful utilities
from ultralytics.data.split import autosplit  # optional
import random

def split():
    # -------- CONFIG ----------
    # IMG_DIR = Path("./datasets/armor_dataset/images")   # 只放图片（或 images/ 子目录）
    # LBL_DIR = Path("./datasets/armor_dataset/labels")   # yolo txt labels parallel to images
    # OUT_DIR = Path("./datasets/splited_data")
    # K = 5

    SEED = 42
    EPOCHS = 30
    PRETRAIN = "yolo12n.pt"  # 初始权重

    # NAMES = ["B1","B2","B3","B4","B5","BG","R1","R2","R3","R4","R5","RG"]  # 替换为你的类别名称列表
    # NAMES = ["car"]  # 替换为你的类别名称列表
    USE_SYMLINK = True  # True: 创建符号链接（节省空间），False: 复制文件
    # ---------------------------

    random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # gather images (only common image ext)
    img_files = sorted([p for p in IMG_DIR.rglob("*") if p.suffix.lower() in ('.jpg','.jpeg','.png')])
    pairs = []
    for img in img_files:
        lbl = (LBL_DIR / (img.stem + ".txt"))
        pairs.append((str(img), str(lbl) if lbl.exists() else None))

    print(f"Found {len(pairs)} images (with label existence noted).")

    # kf = KFold(n_splits=K, shuffle=True, random_state=SEED)
    ss = ShuffleSplit(n_splits=1, test_size=test_size, random_state=SEED)

    fold = 0
    for train_idx, val_idx in ss.split(pairs):
        # fold_dir = OUT_DIR / f"fold{fold}"
        fold_dir = OUT_DIR
        train_img_dir = fold_dir / "train" / "images"
        train_lbl_dir = fold_dir / "train" / "labels"
        val_img_dir = fold_dir / "val" / "images"
        val_lbl_dir = fold_dir / "val" / "labels"
        for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
            d.mkdir(parents=True, exist_ok=True)

        def put_file(src, dst_dir):
            src = Path(src)
            dst = dst_dir / src.name
            if USE_SYMLINK:
                try:
                    if dst.exists(): dst.unlink()
                    os.symlink(os.path.abspath(src), str(dst))
                except Exception:
                    shutil.copy2(src, dst)
            else:
                shutil.copy2(src, dst)

        # populate train
        for i in train_idx:
            img, lbl = pairs[i]
            put_file(img, train_img_dir)
            if lbl and Path(lbl).exists():
                put_file(lbl, train_lbl_dir)
            else:
                print(f"Warning: {img} has no label.")
        # populate val
        for i in val_idx:
            img, lbl = pairs[i]
            put_file(img, val_img_dir)
            if lbl and Path(lbl).exists():
                put_file(lbl, val_lbl_dir)
            else:
                print(f"Warning: {img} has no label.")

        # write data yaml for this fold
        data = {
            'train': "./train/images",
            'val': "./val/images",
            'nc': len(NAMES),
            'names': NAMES
        }
        with open(fold_dir / "data.yaml", "w") as f:
            yaml.dump(data, f)

        # train
        # model = YOLO(PRETRAIN)  # 每折从相同预训练权重开始
        # print(f"Starting training fold {fold} ...")
        # model.train(data=str(fold_dir / "data.yaml"),
        #             epochs=EPOCHS,
        #             project='runs/Kfolder',
        #             name=f"train",
        #             exist_ok=True)
        # val (可选)
        # val_results = model.val(data=str(fold_dir / "data.yaml"))
        # print(f"Fold {fold} validation results:", val_results)

        fold += 1

    print("Finished.")

if __name__=='__main__':
    # 这里可以提供一些参数接口。
    # DIRs,NAMES,PRETRAIN,EPOCH
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--images', help='Path to images directory', type=str, required=True)
    parser.add_argument('-l', '--labels', help='Path to labels directory', type=str, required=True)
    parser.add_argument('-o', '--output', help='Path to output directory', type=str, required=True)
    parser.add_argument('-t', '--test_size', help='Test size', type=float, default=0.2)
    parser.add_argument('-n', '--names', help='Names of classes', type=str, nargs='+', required=True)
    args = parser.parse_args()

    IMG_DIR = Path("./datasets/" + args.images)
    LBL_DIR = Path("./datasets/" + args.labels)
    OUT_DIR = Path("./datasets/" + args.output)
    NAMES = args.names
    test_size = args.test_size
    split()