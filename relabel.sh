python code/labels_reclassify.py --map "[0:2, 1:2, 2:2, 3:2, 4:2, 5:2, 6:1, 7:1, 8:1, 9:1, 10:1, 11:1]" \
-l "armor_dataset/labels" \
-y "armor_dataset_split/data.yaml" \
-n dead red blue
# note that all labels and images in "split" folders are symlinked to the original dataset folders!!!!!