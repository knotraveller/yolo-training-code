# python code/split_data.py -i "car_dataset/images000001-002499" -l "car_dataset/labels000001-053796/labels" -o "car_dataset_split"
# python code/split_data.py -i "car_dataset/images002500-006500" -l "car_dataset/labels000001-053796/labels" -o "car_dataset_split"
# python code/split_data.py -i "car_dataset/images006501-012500" -l "car_dataset/labels000001-053796/labels" -o "car_dataset_split"
# python code/split_data.py -i "car_dataset/images012501-025000" -l "car_dataset/labels000001-053796/labels" -o "car_dataset_split"
# python code/split_data.py -i "car_dataset/images025001-053796" -l "car_dataset/labels000001-053796/labels" -o "car_dataset_split"

# python code/split_data.py -i "armor_dataset/images" -l "armor_dataset/labels" -o "armor_dataset_split"

# python code/split_data.py \
# -i "from_video/7.25_港科大VS新泻大学_BO2_1_0001" \
# -l "from_video/7.25_港科大VS新泻大学_BO2_1_0001/train/labels" \
# -o "from_video/(splited)7.25_港科大VS新泻大学_BO2_1_0001" \
# -n car

python code/split_data.py \
-i "SCAU_car/images" \
-l "SCAU_car/labels" \
-o "SCAU_car_split" \
-n car \ 
-s True