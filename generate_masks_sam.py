from tqdm import tqdm
import torch
import pandas as pd
import cv2
import os
from collections import defaultdict
import numpy as np
from segment_anything import SamPredictor
from segment_anything import sam_model_registry
from PIL import Image
from scipy.spatial import KDTree 
from PIL import Image

def create_image_box_list(csv_path, image_folder, default_box_size=55, small_box_size=35, threshold_distance=7):
    df = pd.read_csv(csv_path)
    image_boxes = defaultdict(list)

    points = list(zip(df['x'], df['y']))
    tree = KDTree(points)

    for index, row in df.iterrows():
        image_path = os.path.join(image_folder, row['images'])
        x, y = row['x'], row['y']


        distances, _ = tree.query((x, y), k=2)  
        nearest_distance = distances[1]  

        box_size = small_box_size if nearest_distance < threshold_distance else default_box_size

        x1 = max(0, x - int(0.55 * box_size))
        x2 = min(x + int(0.55 * box_size), 2048)

        y1 = max(0, y - int(0.6 * box_size))
        y2 = min(y + int(0.4 * box_size), 2048)

        image_boxes[image_path].append((x1, y1, x2, y2))

    return image_boxes

def or_binary_maps(maps):
    if not maps:
        raise ValueError("Input list of maps is empty.")

    result_map = np.zeros_like(maps[0], dtype=bool)

    for map_ in maps:
        result_map |= map_

    return result_map

def generate(args):

    if os.path.exists(args.masks_path) and len(os.listdir(args.masks_path)) > 0:
        print(f"Skipping mask generation. Masks already exist in {args.masks_path}.")
        return  

    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    MODEL_TYPE = "vit_h"

    sam = sam_model_registry[MODEL_TYPE](checkpoint=args.sam_path)
    sam = sam.to(device=DEVICE)

    csv_path = args.csv_file_dir
    image_folder = args.imag_non_empty_dir

    mask_predictor = SamPredictor(sam)

    image_box_dict = create_image_box_list(csv_path, image_folder, default_box_size=55, small_box_size=35, threshold_distance=7)

    os.makedirs(args.masks_path, exist_ok=True)

    for image_path, boxes in tqdm(image_box_dict.items(), desc="Processing Images"):
        image_bgr = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mask_predictor.set_image(image_rgb)
        list_of_masks = []

        for b in boxes:
            box = np.array(b)
            masks, scores, logits = mask_predictor.predict(box=box, multimask_output=True)

            valid_masks = [m for m in masks if np.sum(m) > 300]  

            if valid_masks:
                mask = np.logical_or.reduce(valid_masks)  
            else:
                mask = masks[0]  

            list_of_masks.append(mask)

        mask = or_binary_maps(list_of_masks)
        mask = np.clip(mask, 0, 1)


        mask_filename = os.path.basename(image_path)
        mask_path = os.path.join(args.masks_path, mask_filename)
        mask_image = Image.fromarray((mask * 255).astype(np.uint8))
        mask_image.save(mask_path)

    print(f"Masks saved successfully to {args.masks_path}.")
