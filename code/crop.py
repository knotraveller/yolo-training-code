from pathlib import Path
from PIL import Image
import argparse
from tqdm import tqdm


def find_files(root_path, extensions):
    root = Path(root_path)
    files = []
    for ext in extensions:
        files.extend(root.rglob(f"*{ext}"))
    return files


def yolo_to_bbox(x_center, y_center, w, h, img_width, img_height):
    """将YOLO格式转换为PIL的bbox格式"""
    x_center_abs = x_center * img_width
    y_center_abs = y_center * img_height
    w_abs = w * img_width
    h_abs = h * img_height
    
    left = int(x_center_abs - w_abs / 2)
    top = int(y_center_abs - h_abs / 2)
    right = int(x_center_abs + w_abs / 2)
    bottom = int(y_center_abs + h_abs / 2)
    
    # 虽然估计没必要，但姑且写一下
    left = max(0, left)
    top = max(0, top)
    right = min(img_width, right)
    bottom = min(img_height, bottom)
    
    return (left, top, right, bottom)


def process_dataset(images_path, labels_path, output_path):
    output_images = Path(output_path) / "images"
    output_labels = Path(output_path) / "labels"
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)


    image_files = find_files(images_path, ['.jpg', '.jpeg', '.png', '.bmp'])
    label_files = find_files(labels_path, ['.txt'])
    
    # 创建标签文件映射 (stem -> path)
    label_dict = {lf.stem: lf for lf in label_files}
    
    print(f"找到 {len(image_files)} 个图片文件")
    print(f"找到 {len(label_files)} 个标签文件")
    
    processed_count = 0
    cropped_count = 0
    
    for img_path in tqdm(image_files):
        # 查找
        label_path = label_dict.get(img_path.stem)
        
        if label_path is None:
            print(f"未找到 {img_path.name} 对应的标签文件")
            continue
        
        # 读图片
        try:
            img = Image.open(img_path)
            img_width, img_height = img.size
        except Exception as e:
            print(f"无法读取图片 {img_path.name}: {e}")
            continue
        
        # 读标签
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"无法读取标签 {label_path.name}: {e}")
            continue
        
        # 处理
        for idx, line in enumerate(lines, start=1):
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            cls, x, y, w, h = parts[0], float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            
            # 转换坐标
            bbox = yolo_to_bbox(x, y, w, h, img_width, img_height)
            cropped_img = img.crop(bbox)
            
            # 保存
            cropped_img_name = f"{img_path.stem}_{idx:02d}{img_path.suffix}"
            cropped_img_path = output_images / cropped_img_name
            cropped_img.save(cropped_img_path)
            
            label_name = f"{img_path.stem}_{idx:02d}.txt"
            label_out_path = output_labels / label_name
            with open(label_out_path, 'w') as f:
                f.write(cls)
            
            cropped_count += 1
        
        processed_count += 1
        # if processed_count % 100 == 0:
        #     print(f"已处理 {processed_count} 个图片...")
    
    print(f"\nComplete!")
    print(f"处理图片数: {processed_count}")
    print(f"裁剪区域数: {cropped_count}")
    print(f"Save to: {output_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="从YOLO格式标签裁剪图片区域",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python crop.py --images ./data/images --labels ./data/labels --output ./output
  python crop.py -i ./images -l ./labels -o ./cropped
        """
    )
    
    parser.add_argument('-i', '--images', help='Path to images directory', type=str, required=True)
    parser.add_argument('-l', '--labels', help='Path to labels directory', type=str, required=True)
    parser.add_argument('-o', '--output', help='Path to output directory', type=str, required=True)
    
    args = parser.parse_args()
    
    process_dataset(args.images, args.labels, args.output)
