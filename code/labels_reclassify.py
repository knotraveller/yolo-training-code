import shutil
import os
from pathlib import Path
import yaml
import argparse
import re
import datetime

def parse_map_string(map_str: str):
    """
    解析映射字符串，返回 dict[int->int]
    支持格式示例: "[10:3, 5:7]" 或 "10:3,5:7" 或 "{10:3,5:7}"
    """
    if not isinstance(map_str, str):
        raise ValueError("映射必须为字符串，例如 \"[10:3, 5:7]\"")
    s = map_str.strip()
    # 去掉外层括号或花括号
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1]
    pairs = [p.strip() for p in s.split(",") if p.strip()]
    mapping = {}
    for p in pairs:
        # 支持 "a:b" 或 "a -> b"
        if ":" in p:
            left, right = p.split(":", 1)
        elif "->" in p:
            left, right = p.split("->", 1)
        elif "-" in p and p.count("-") == 1:
            left, right = p.split("-", 1)
        else:
            raise ValueError(f"无法解析映射片段: '{p}'，请使用 ':' 或 '->' 分隔")
        left = left.strip()
        right = right.strip()
        mapping[left] = right
    return mapping

def parse_names_string(names_str: str):
    """
    解析 names 字符串，例如 "[car, bike]" 或 "car,bike" 或 "['car','bike']"
    返回 list[str]
    """
    if names_str is None:
        return None
    s = names_str.strip()
    # 去掉外层括号或花括号
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1]
    # 切分 by comma
    parts = [p.strip() for p in s.split(",") if p.strip()]
    # 去掉可能的引号
    cleaned = []
    for p in parts:
        # if (p.startswith("'") and p.endswith("'")) or (p.startswith('"') and p.endswith('"')):
        #     p = p[1:-1]
        cleaned.append(p)
    return cleaned


def find_single_yaml(start_dir: Path):
    """
    在 start_dir 及其父目录中查找唯一的 .yaml/.yml 文件
    返回 Path 或 None（若没找到或找到多个）
    """
    candidates = []
    for d in [start_dir, start_dir.parent]:
        if not d.exists():
            continue
        for p in d.glob("*.yaml"):
            candidates.append(p)
        for p in d.glob("*.yml"):
            candidates.append(p)
    # 去重并保持唯一
    candidates = sorted(set(candidates))
    if len(candidates) == 1:
        return candidates[0]
    return None

def remap_label_file(src_path: Path, dst_path: Path, mapping: dict):
    """
    读取 src_path 的 yolo txt，替换每行第一个类索引，保存到 dst_path。
    保持行格式其余不变（支持空行）
    """
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with src_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()
    out_lines = []
    for ln in lines:
        s = ln.strip()
        if s == "":
            out_lines.append("\n")
            continue
        parts = s.split()
        clss = parts[0]
        new_cls = mapping.get(clss, clss)  # 若未映射则保持原值
        parts[0] = str(new_cls)
        out_lines.append(" ".join(parts) + "\n")
    with dst_path.open("w", encoding="utf-8") as f:
        f.writelines(out_lines)

def update_yaml_classes(yaml_path: Path, mapping: dict, names_list):
    """
    读取 yaml，备份为 XXX(old).yaml，然后更新 names 与 nc。
    """
    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # 备份原 yaml 为 XXX(old).yaml
    bak_name = yaml_path.with_name(yaml_path.stem + "(old)" + yaml_path.suffix)
    if bak_name.exists():
        # 若已存在备份，在其文件名后加时间戳以免覆盖
        ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        bak_name = yaml_path.with_name(yaml_path.stem + f"(old){ts}" + yaml_path.suffix)
    shutil.copy2(str(yaml_path), str(bak_name))
    print(f"原 yaml 已备份为: {bak_name}")

    # update data dict
    data['names'] = names_list
    data['nc'] = len(names_list)

    # write back to original yaml path (覆盖原文件)
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False, allow_unicode=True)
    print(f"已写回新的 yaml（原文件名被替换）: {yaml_path}，并将原文件备份到 {bak_name}")
    return data

def remap_labels_folder(lbl_dir: Path, mapping: dict):
    """
    将 lbl_dir 重命名为 old_labels（位于同一父目录下），在原位置创建新的 labels 文件夹并保存修改后的标签。
    保持原相对结构（递归处理 .txt）
    """
    if not lbl_dir.exists() or not lbl_dir.is_dir():
        raise FileNotFoundError(f"labels 文件夹不存在: {lbl_dir}")

    parent = lbl_dir.parent
    old_dir = parent / "old_labels"
    if old_dir.exists():
        raise FileExistsError(f"目标备份文件夹已存在: {old_dir}。请先移除或手动重命名。脚本不会覆盖已存在的 old_labels。")
    # 重命名原来文件夹为 old_labels
    lbl_dir.rename(old_dir)
    print(f"已将原 labels 目录重命名为: {old_dir}")

    # 新的 labels 目录（同名）
    new_dir = parent / "labels"
    new_dir.mkdir(parents=True, exist_ok=True)

    # 递归处理所有 txt 文件
    txt_files = list(old_dir.rglob("*.txt"))
    print(f"发现 {len(txt_files)} 个 txt 标签文件，开始重映射...")
    for src in txt_files:
        # 目标相对路径保留
        rel = src.relative_to(old_dir)
        dst = new_dir / rel
        remap_label_file(src, dst, mapping)
    # 也要保留其他非 .txt 文件夹/文件的结构（可选）；这里只处理 txt
    print(f"映射完成，新的 labels 存放在: {new_dir}")

    return old_dir, new_dir

def main():
    parser = argparse.ArgumentParser(description="批量 remap YOLO labels 的 class id 并更新 yaml")
    parser.add_argument("--map", "-m", required=True, help='映射字符串，例如 "[10:3, 5:7]"（old:new）')
    parser.add_argument("--labels", "-l", required=True, help="labels 文件夹路径（yolo txt 所在目录）")
    parser.add_argument("--data-yaml", "-y", required=False, default=None, help="可选：指定 data.yaml 路径（若不指定脚本会在 labels 的父目录和祖父目录尝试自动定位）")
    parser.add_argument("--names", "-n", required=False, default=None, nargs="+", help='可选：用一个 names 列表覆盖 yaml，例如 B1 B2 ')
    args = parser.parse_args()

    mapping = parse_map_string(args.map)
    LBL_DIR = os.path.join('./datasets', args.labels)
    LBL_DIR = Path(LBL_DIR).expanduser().resolve()

    # step1: remap labels folder
    old_dir, new_dir = remap_labels_folder(LBL_DIR, mapping)

    # step2: locate yaml
    yaml_path = None
    if args.data_yaml:
        yaml_path = os.path.join('./datasets', args.data_yaml)
        yaml_path = Path(yaml_path).expanduser().resolve()
        if not yaml_path.exists():
            raise FileNotFoundError(f"指定的 yaml 文件不存在: {yaml_path}")
    else:
        detected = find_single_yaml(LBL_DIR)
        if detected:
            yaml_path = detected
            print(f"自动检测到 yaml: {yaml_path}")
        else:
            raise FileNotFoundError("未能自动定位唯一的 yaml 文件。请通过 --data-yaml 指定 data.yaml 的路径。")

    # step3: update yaml
    names_list = args.names
    if names_list:
        # names_list = parse_names_string(names_str)
        new_data = update_yaml_classes(yaml_path, mapping, names_list)  

    print("全部完成。总结：")
    print(f"  原 labels 已重命名为: {old_dir}")
    print(f"  新 labels 已创建为: {new_dir}")
    if names_list:
        print(f"  yaml 已更新并将原文件备份为: {yaml_path.with_name(yaml_path.stem + '(old)' + yaml_path.suffix)}")
    print("注意：如果你的 labels 路径或 yaml 结构比较特殊（例如 names 为自定义格式），请先备份并谨慎使用。")

if __name__ == "__main__":
    main()
