import os, cv2
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from utils import mask_exists
import numpy as np
import torch
import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
from utils import mask_exists  


class TestDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transforms_img, transforms_mask,
                 verify_pixels: bool = True):
        """
        Args:
            image_dir (str): Directory of images.
            mask_dir (str): Directory of masks (same basenames expected).
            transforms_img: Transform pipeline for images.
            transforms_mask: Transform pipeline for masks.
            verify_pixels (bool): If True, open mask & confirm FG pixels > 0
                                  (after resolving mask filename).
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform_img = transforms_img
        self.transform_mask = transforms_mask
        self.verify_pixels = verify_pixels

        self.image_paths = sorted([
            os.path.join(image_dir, fn)
            for fn in os.listdir(image_dir)
            if fn.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

        # print(f"Image directory: {image_dir}")
        # print(f"Mask directory:  {mask_dir}")
        # print(f"# images:        {len(self.image_paths)}")

        self.is_abnormal = []
        self._resolved_mask_paths = []  

        for img_path in self.image_paths:
            basename = os.path.basename(img_path)

            util_exists = mask_exists(basename, self.mask_dir)


            mask_filename_jpg = os.path.splitext(basename)[0] + ".jpg"
            mask_path_jpg = os.path.join(self.mask_dir, mask_filename_jpg)

            mask_path_sameext = os.path.join(self.mask_dir, basename)

            if util_exists and os.path.exists(mask_path_jpg):
                use_path = mask_path_jpg
            elif os.path.exists(mask_path_sameext):
                use_path = mask_path_sameext
            else:
                use_path = None  

            if use_path is None:
                self.is_abnormal.append(False)
                self._resolved_mask_paths.append(None)
                continue

            if self.verify_pixels:
                try:
                    m = Image.open(use_path).convert("L")
                    m_arr = np.array(m)
                    ab = bool(np.any(m_arr > 0))
                except Exception as e:
                    print(f"[warn] failed to load mask {use_path}: {e}; marking normal.")
                    ab = False
            else:
                ab = True  

            self.is_abnormal.append(ab)
            self._resolved_mask_paths.append(use_path if ab else None)

        n_ab = sum(self.is_abnormal)
        print(f"Found {n_ab} abnormal / {len(self.is_abnormal) - n_ab} normal patches in validation set.")


    def __len__(self):
        return len(self.image_paths)


    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")

        mask_path = self._resolved_mask_paths[idx]
        if mask_path is not None and self.is_abnormal[idx]:
            try:
                mask = Image.open(mask_path).convert("L")
            except Exception as e:
                print(f"[warn] failed to load mask {mask_path} at __getitem__: {e}")
                w, h = img.size
                mask = Image.fromarray(np.zeros((h, w), dtype=np.uint8))
        else:
            w, h = img.size
            mask = Image.fromarray(np.zeros((h, w), dtype=np.uint8))

        mask_np = np.array(mask)
        _, mask_np = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
        mask      = Image.fromarray(mask_np)

        mask_exists_tensor = torch.tensor(
            1.0 if self.is_abnormal[idx] else 0.0,
            dtype=torch.float32
        )

        img  = self.transform_img(img)
        
        mask = self.transform_mask(mask)
        return img, mask, img_path, mask_exists_tensor
