# python code/split_data.py -i "car_dataset/images000001-002499" -l "car_dataset/labels000001-053796/labels" -o "car_dataset_split"
# python code/split_data.py -i "car_dataset/images002500-006500" -l "car_dataset/labels000001-053796/labels" -o "car_dataset_split"
# python code/split_data.py -i "car_dataset/images006501-012500" -l "car_dataset/labels000001-053796/labels" -o "car_dataset_split"
# python code/split_data.py -i "car_dataset/images012501-025000" -l "car_dataset/labels000001-053796/labels" -o "car_dataset_split"
# python code/split_data.py -i "car_dataset/images025001-053796" -l "car_dataset/labels000001-053796/labels" -o "car_dataset_split"

# python code/split_data.py -i "armor_dataset/images" -l "armor_dataset/labels" -o "armor_dataset_split"

python code/split_data.py \
-i "RM2025-Armor-Public-Dataset" \
-l "RM2025-Armor-Public-Dataset" \
-o "RM2025-Armor-Public-Dataset_split" \
-n dead red blue 