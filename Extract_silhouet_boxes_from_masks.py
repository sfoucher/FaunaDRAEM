# Extract silhoutette and mask boxes from bounding boxes containing animals
import os
from pathlib import Path
import numpy as np
from skimage import io, measure
from collections import defaultdict
from tqdm import tqdm

def extract_silhouettes_and_masks(mask_dir, patch_dir,
                                  silhouette_output_dir, mask_output_dir,
                                  min_size: int = 20,
                                  padding: int = 10,    
                                  PREFIX_TOKENS: int = 3):
    """
    Extract silhouette and corresponding mask bounding boxes from patches and masks.
    Save silhouettes in silhouette_output_dir and masks in mask_output_dir with matching IDs.

    Parameters
    ----------
    mask_dir : str or Path
        Directory containing binary mask images.
    patch_dir : str or Path
        Directory containing RGB patch images.
    silhouette_output_dir : str or Path
        Directory to save cropped silhouette images.
    mask_output_dir : str or Path
        Directory to save cropped mask images.
    min_size : int, default 20
        Minimum connected component size to keep.
    padding : int, default 45
        Pixels padding around bounding boxes.
    PREFIX_TOKENS : int, default 3
        Number of underscore-separated tokens to define image prefix.
    """
    mask_dir = Path(mask_dir)
    patch_dir = Path(patch_dir)
    silhouette_output_dir = Path(silhouette_output_dir)
    mask_output_dir = Path(mask_output_dir)

    silhouette_output_dir.mkdir(parents=True, exist_ok=True)
    mask_output_dir.mkdir(parents=True, exist_ok=True)

    mask_files = [f for f in os.listdir(mask_dir) if f.lower().endswith((".jpg", ".png"))]

    counters = defaultdict(int)  

    for mask_file in tqdm(mask_files, desc="Extracting silhouettes and masks"):
        mask_path = mask_dir / mask_file
        patch_path = patch_dir / mask_file 

        if not patch_path.exists():
            print(f"Patch file not found for {mask_file}, skipping.")
            continue

        base_name = Path(mask_file).stem
        tokens = base_name.split("_")
        img_prefix = "_".join(tokens[:PREFIX_TOKENS]) 
        patch_id = base_name

        mask = io.imread(mask_path, as_gray=True)
        patch = io.imread(patch_path)

        binary_mask = (mask > 0).astype(np.uint8)
        labeled_mask = measure.label(binary_mask, connectivity=2)

        num_regions = 0
        for region in measure.regionprops(labeled_mask):
            if region.area < min_size:
                continue

            minr, minc, maxr, maxc = region.bbox
            original_height = maxr - minr
            original_width = maxc - minc
            padded_minr = max(0, minr - padding)
            padded_minc = max(0, minc - padding)
            padded_maxr = min(mask.shape[0], maxr + padding)
            padded_maxc = min(mask.shape[1], maxc + padding)
            padded_height = padded_maxr - padded_minr
            padded_width = padded_maxc - padded_minc

            silhouette = patch[padded_minr:padded_maxr, padded_minc:padded_maxc]
            mask_crop = mask[padded_minr:padded_maxr, padded_minc:padded_maxc]  # Recadrer le masque original

            sil_idx = counters[img_prefix]
            counters[img_prefix] += 1

            sil_name = f"{patch_id}_sil{sil_idx:04d}.png"
            mask_name = f"{patch_id}_mask{sil_idx:04d}.png"

            io.imsave(silhouette_output_dir / sil_name, silhouette)
            io.imsave(mask_output_dir / mask_name, mask_crop.astype(np.uint8))

            print(f"[{mask_file}] Région {sil_idx}: "
                  f"bbox original HxW=({original_height}x{original_width}), "
                  f"bbox paddé HxW=({padded_height}x{padded_width})")

            num_regions += 1

        print(f"--> {num_regions} silhouettes extraites dans {mask_file}")

    total = sum(counters.values())
    print(f"Finished: {total} silhouette and mask boxes saved.")

patch_dir = '/DRAEM/DATASETS/Multi_Herd_data/Multi_Herd_allocation/Multi_Herd_Patches/NOM_Patches/TRAIN_NOM_NonEmpty'
mask_dir = '/DRAEM/DATASETS/Multi_Herd_data/TRAIN_SAM_MASKS'
silhouette_output_dir = "/DRAEM/Silhoute_boxes_Multiherd/Silhouettes"
mask_output_dir = "/DRAEM/Silhoute_boxes_Multiherd/Masks"

extract_silhouettes_and_masks(mask_dir, patch_dir,
                              silhouette_output_dir, mask_output_dir,
                              padding=10)

