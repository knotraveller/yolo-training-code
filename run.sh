# python code/main.py --train --val --predict -d "Car Data" -m yolo12n.pt
# python code/main.py --train -d "Car Data" -m 'yolo12n.pt' -e 1
# python code/main.py --predict -d "armor_dataset(part)/images" -m './model/armor.onnx'
# python code/main.py -m "./runs/Kfolder/train0/weights/best.pt" -d "splited_data/fold0/val/images" --predict

# python code/main.py --train --val --predict -d "car_dataset_split" -m "yolo12m.pt" -e 100
# python code/main.py --train --resume --val -d "car_dataset_split" -m "runs/train/yolo12m_car_dataset_split/weights/last.pt"
python code/main.py --train --val -d "Armor" -m "yolo12m.pt" -e 100