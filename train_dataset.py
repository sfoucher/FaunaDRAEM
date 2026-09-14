import random
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import imgaug.augmenters as iaa
import numpy as np
import torch
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from torch.utils.data import Dataset


def edge_aware_feather(
    mask_bin: np.ndarray,
    feather_width: int = 5,
) -> np.ndarray:
    """Create an inward edge-aware alpha mask."""
    mask = (mask_bin > 127).astype(np.uint8) * 255

    distance_inside = cv2.distanceTransform(
        mask,
        cv2.DIST_L2,
        5,
    )

    feather_width_float = float(
        max(1, feather_width)
    )

    alpha = np.clip(
        distance_inside / feather_width_float,
        0.0,
        1.0,
    )

    alpha[mask == 0] = 0.0

    alpha[
        distance_inside
        > feather_width_float + 2.0
    ] = 1.0

    return alpha.astype(np.float32)


class TrainDataset(Dataset):
    def __init__(
        self,
        bg_dir: str,
        sil_root: str,
        mask_root: str,
        resize: Tuple[int, int],
        transforms_img,
        save_reconstructed: bool,
        transforms_mask,
        max_silhouettes: int = 80,
        epoch: int = -1,
        batch_size: int = 8,
        # augmentations other than scaling are optional.
        apply_aug: bool = False,
        is_last_epoch: bool = False,
        save_synthetic: bool = False,
        mode: str = "both",
        p_clean: float = 0.5,
        p_geo: float = 0.5,
        p_flip: float = 0.3,
        p_photo: float = 0.2,
        cache_silhouettes: bool = True,
    ):
        super().__init__()

        self.apply_aug = bool(apply_aug)

        self.p_geo = float(p_geo)
        self.p_flip = float(p_flip)
        self.p_photo = float(p_photo)

        # Scaling stays independent from apply_aug.
        self.p_scale = 0.40
        self.scale_range = (1.05, 1.30)

        self._aug_geo_count = 0
        self._aug_photo_count = 0
        self._aug_scale_count = 0
        self._getitem_calls = 0
        self.cache_silhouettes = bool(cache_silhouettes)

        self._silhouette_cache: Dict[
            Tuple[str, str],
            Tuple[np.ndarray, np.ndarray],
        ] = {}
        self.batch_size = int(batch_size)
        self.is_last_epoch = bool(is_last_epoch)
        self.mode = str(mode)
        self.p_clean = float(p_clean)
        self.save_reconstructed = bool(
            save_reconstructed
        )
        self.epoch = epoch

        self.bg_paths: List[Path] = (
            sorted(
                Path(bg_dir).glob("*.jpg")
            )
            + sorted(
                Path(bg_dir).glob("*.png")
            )
        )

        if not self.bg_paths:
            raise RuntimeError(
                f"No background images found in {bg_dir}"
            )

        self.group_pairs: Dict[
            Path,
            List[Tuple[Path, Path]],
        ] = {}

        for group in Path(sil_root).iterdir():
            if not group.is_dir():
                continue

            silhouette_dir = (
                group / "sil_boxes"
            )

            group_mask_dir = (
                group / "masks"
            )

            if (
                not silhouette_dir.is_dir()
                or not group_mask_dir.is_dir()
            ):
                continue

            pairs = [
                (
                    silhouette_path,
                    group_mask_dir
                    / silhouette_path.name,
                )
                for silhouette_path
                in silhouette_dir.glob("*.png")
                if (
                    group_mask_dir
                    / silhouette_path.name
                ).exists()
            ]

            if pairs:
                self.group_pairs[
                    group
                ] = pairs

        if not self.group_pairs:
            silhouette_dir = (
                Path(sil_root)
                / "sil_boxes"
            )

            flat_mask_dir = (
                Path(mask_root)
                / "masks"
            )

            if (
                silhouette_dir.is_dir()
                and flat_mask_dir.is_dir()
            ):
                pairs = [
                    (
                        silhouette_path,
                        flat_mask_dir
                        / silhouette_path.name,
                    )
                    for silhouette_path
                    in silhouette_dir.glob(
                        "*.png"
                    )
                    if (
                        flat_mask_dir
                        / silhouette_path.name
                    ).exists()
                ]

                if pairs:
                    self.group_pairs[
                        Path(sil_root)
                    ] = pairs

        if not self.group_pairs:
            raise RuntimeError(
                "No silhouette/mask groups found."
            )

        self.group_dirs = list(
            self.group_pairs.keys()
        )

        self.resize = resize
        self.max_silhouettes = int(
            max_silhouettes
        )

        self.tr_img = transforms_img
        self.tr_msk = transforms_mask

        self.save_synthetic = bool(
            save_synthetic
        )

        self.debug_save_every = 400
        self.debug_epoch_only = None

        self.dbg_root = Path(
            "/DRAEM/Paper_3_RESULTS/"
            "Synthetic_patches_3"
        )

        if (
            self.save_synthetic
            and self.debug_save_every
        ):
            self.dbg_root.mkdir(
                parents=True,
                exist_ok=True,
            )

        # print(
        #     "[TrainDataset] scaling active independently: "
        #     f"p_scale={self.p_scale}, "
        #     f"range={self.scale_range}"
        # )

        if self.apply_aug:
            # print(
            #     "[TrainDataset] optional augmentations ENABLED: "
            #     f"p_geo={self.p_geo}, "
            #     f"p_flip={self.p_flip}, "
            #     f"p_photo={self.p_photo}"
            # )

            self.geo_seq = iaa.Sequential(
                [
                    iaa.Sometimes(
                        self.p_flip,
                        iaa.Fliplr(1.0),
                    ),
                ]
            )

            # Photometric augmentation is applied only to RGB.
            self.photo_seq = iaa.Sequential(
                [
                    iaa.AddToBrightness(
                        add=(-10, 10)
                    ),
                    iaa.GammaContrast(
                        (0.9, 1.1)
                    ),
                ],
                random_order=True,
            )
        else:
            print(
                "[TrainDataset] optional augmentations DISABLED"
            )

            # No augmentation objects are used in __getitem__.
            self.geo_seq = None
            self.photo_seq = None
    
    def __len__(self) -> int:
        return len(self.bg_paths)

    def set_debug_epoch(
        self,
        epoch: int,
    ) -> None:
        self.debug_epoch_only = epoch

    def _sample_scale(self) -> float:
        """Sample enlargement without shrinking."""
        low, high = self.scale_range

        return float(
            np.random.uniform(
                low,
                high,
            )
        )

    def _adaptive_feather_width(
        self,
        height: int,
        width: int,
    ) -> int:
        """Preserve contrast for small silhouettes."""
        minimum_dimension = min(
            height,
            width,
        )

        if minimum_dimension <= 18:
            return 1

        if minimum_dimension <= 28:
            return 2

        return 4

    def _make_clean(
        self,
        bg_rgb: np.ndarray,
        height: int,
        width: int,
    ):
        resized = cv2.resize(
            bg_rgb,
            (width, height),
            interpolation=cv2.INTER_AREA,
        )

        clean_img = self.tr_img(
            resized
        )

        zero_mask = np.zeros(
            (height, width),
            dtype=np.uint8,
        )

        mask_tensor = self.tr_msk(
            zero_mask
        )

        synth_img = clean_img
        mask_exists = torch.tensor(
            0.0,
            dtype=torch.float32,
        )

        return (
            clean_img,
            mask_tensor,
            synth_img,
            mask_exists,
        )
    
    def _load_silhouette_pair(
        self,
        silhouette_path: Path,
        mask_path: Path,
    ) -> Tuple[
        np.ndarray | None,
        np.ndarray | None,
    ]:
        """Load one silhouette/mask pair and reuse it from RAM."""
        cache_key = (
            str(silhouette_path),
            str(mask_path),
        )

        if (
            self.cache_silhouettes
            and cache_key in self._silhouette_cache
        ):
            source_rgb, mask = self._silhouette_cache[cache_key]

            if self.apply_aug:
                return (
                    source_rgb.copy(),
                    mask.copy(),
                )

            return source_rgb, mask

        # Cache miss: read from disk once.
        source_bgr = cv2.imread(
            str(silhouette_path),
            cv2.IMREAD_COLOR,
        )

        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if source_bgr is None or mask is None:
            return None, None

        source_rgb = cv2.cvtColor(
            source_bgr,
            cv2.COLOR_BGR2RGB,
        )

        if self.cache_silhouettes:
            self._silhouette_cache[cache_key] = (
                source_rgb,
                mask,
            )

        if self.apply_aug:
            return (
                source_rgb.copy(),
                mask.copy(),
            )

        return source_rgb, mask

    
    def __getitem__(
        self,
        idx: int,
    ):
        height, width = self.resize
        pad = 100

        background_bgr = cv2.imread(
            str(
                self.bg_paths[idx]
            )
        )

        if background_bgr is None:
            raise RuntimeError(
                "Failed to read background: "
                f"{self.bg_paths[idx]}"
            )

        if (
            background_bgr.shape[:2]
            != (512, 512)
        ):
            raise ValueError(
                "Expected 512x512 background, "
                f"got {background_bgr.shape[:2]}"
            )

        background_rgb = cv2.cvtColor(
            background_bgr,
            cv2.COLOR_BGR2RGB,
        )

        if self.mode == "clean":
            make_empty = True
        elif self.mode == "anom":
            make_empty = False
        elif self.mode == "both":
            make_empty = (
                random.random()
                < self.p_clean
            )
        else:
            raise ValueError(
                "mode must be one of "
                "'clean', 'anom', or 'both'."
            )

        if make_empty:
            return self._make_clean(
                background_rgb,
                height,
                width,
            )

        canvas_height = (
            512 + 2 * pad
        )

        canvas_width = (
            512 + 2 * pad
        )

        canvas = np.full(
            (
                canvas_height,
                canvas_width,
                3,
            ),
            int(
                background_rgb.mean()
            ),
            dtype=np.uint8,
        )

        canvas[
            pad:pad + 512,
            pad:pad + 512,
        ] = background_rgb

        mask_canvas = np.zeros(
            (
                canvas_height,
                canvas_width,
            ),
            dtype=np.uint8,
        )

        placed: List[
            Tuple[int, int, int, int]
        ] = []

        group = random.choice(
            self.group_dirs
        )

        pairs = self.group_pairs[
            group
        ]

        # ---------------------------------------------------------
        # Scaling is independent from apply_aug.
        # ---------------------------------------------------------
        do_scale_shared = (
            random.random()
            < self.p_scale
        )

        shared_scale = (
            self._sample_scale()
            if do_scale_shared
            else 1.0
        )

        
        silhouette_count = random.randint(
            1,
            self.max_silhouettes,
        )

        if len(pairs) >= silhouette_count:
            selected = random.sample(
                pairs,
                silhouette_count,
            )
        else:
            selected = random.choices(
                pairs,
                k=silhouette_count,
            )

        applied_geo_any = False
        applied_photo_any = False

        for (
            silhouette_path,
            mask_path,
        ) in selected:
            source_rgb, mask = self._load_silhouette_pair(
                silhouette_path,
                mask_path,
            )

            if source_rgb is None or mask is None:
                continue

            if (
                self.geo_seq is not None
                and random.random() < self.p_geo
            ):
                applied_geo_any = True

                deterministic_geo = (
                    self.geo_seq.to_deterministic()
                )

                seg_map = SegmentationMapsOnImage(
                    (mask > 127).astype(np.uint8),
                    shape=source_rgb.shape,
                )

                source_rgb = deterministic_geo(
                    image=source_rgb,
                )

                seg_map = deterministic_geo(
                    segmentation_maps=seg_map,
                )

                mask = (
                    (seg_map.get_arr() > 0)
                    .astype(np.uint8)
                    * 255
                )

            if (
                self.photo_seq is not None
                and random.random() < self.p_photo
            ):
                applied_photo_any = True

                source_rgb = self.photo_seq(
                    image=source_rgb,
                )

            _, binary_mask = cv2.threshold(
                mask,
                127,
                255,
                cv2.THRESH_BINARY,
            )


            nonzero = cv2.findNonZero(
                binary_mask
            )

            if nonzero is None:
                continue

            (
                x,
                y,
                bbox_width,
                bbox_height,
            ) = cv2.boundingRect(
                nonzero
            )

            crop = source_rgb[
                y:y + bbox_height,
                x:x + bbox_width,
            ]

            binary_mask = binary_mask[
                y:y + bbox_height,
                x:x + bbox_width,
            ]


            # -----------------------------------------------------
            # Shared enlargement.
            # Independent from apply_aug.
            # -----------------------------------------------------
            if shared_scale != 1.0:
                crop = cv2.resize(
                    crop,
                    dsize=None,
                    fx=shared_scale,
                    fy=shared_scale,
                    interpolation=cv2.INTER_CUBIC,
                )

                binary_mask = (
                    cv2.resize(
                        binary_mask,
                        dsize=None,
                        fx=shared_scale,
                        fy=shared_scale,
                        interpolation=(
                            cv2.INTER_NEAREST
                        ),
                    )
                )

            (
                object_height,
                object_width,
            ) = crop.shape[:2]

            final_x1 = pad
            final_y1 = pad
            final_x2 = pad + 512
            final_y2 = pad + 512

            is_first_visible_object = len(placed) == 0

            if (
                is_first_visible_object
                and object_width <= 512
                and object_height <= 512
            ):
                x1 = random.randint(
                    final_x1,
                    final_x2 - object_width,
                )

                y1 = random.randint(
                    final_y1,
                    final_y2 - object_height,
                )

                x2 = x1 + object_width
                y2 = y1 + object_height

            else:
                placement_found = False

                for _ in range(60):
                    x1 = random.randint(
                        0,
                        canvas_width - object_width,
                    )

                    y1 = random.randint(
                        0,
                        canvas_height - object_height,
                    )

                    x2 = x1 + object_width
                    y2 = y1 + object_height

                    overlap_x1 = max(
                        x1,
                        final_x1,
                    )

                    overlap_y1 = max(
                        y1,
                        final_y1,
                    )

                    overlap_x2 = min(
                        x2,
                        final_x2,
                    )

                    overlap_y2 = min(
                        y2,
                        final_y2,
                    )

                    if (
                        overlap_x1 >= overlap_x2
                        or overlap_y1 >= overlap_y2
                    ):
                        continue

                    # Check the ACTUAL foreground mask, not only its bbox.
                    mask_x1 = overlap_x1 - x1
                    mask_y1 = overlap_y1 - y1
                    mask_x2 = overlap_x2 - x1
                    mask_y2 = overlap_y2 - y1

                    visible_mask = binary_mask[
                        mask_y1:mask_y2,
                        mask_x1:mask_x2,
                    ]

                    if not np.any(
                        visible_mask > 0
                    ):
                        continue

                    no_overlap = all(
                        x2 <= bx1
                        or bx2 <= x1
                        or y2 <= by1
                        or by2 <= y1
                        for (
                            bx1,
                            by1,
                            bx2,
                            by2,
                        ) in placed
                    )

                    if not no_overlap:
                        continue

                    placement_found = True
                    break

                if not placement_found:
                    continue

            feather_width = (
                self._adaptive_feather_width(
                    object_height,
                    object_width,
                )
            )

            alpha = (
                edge_aware_feather(
                    binary_mask,
                    feather_width=(
                        feather_width
                    ),
                )
            )

            alpha_rgb = np.repeat(
                alpha[..., None],
                3,
                axis=2,
            )

            roi = canvas[
                y1:y2,
                x1:x2,
            ].astype(
                np.float32
            )

            foreground = (
                crop.astype(
                    np.float32
                )
            )

            blended = (
                foreground
                * alpha_rgb
                + roi
                * (
                    1.0
                    - alpha_rgb
                )
            )

            canvas[
                y1:y2,
                x1:x2,
            ] = np.clip(
                blended,
                0,
                255,
            ).astype(
                np.uint8
            )

            hard_mask = (
                (
                    alpha
                    > 0.05
                )
                .astype(
                    np.uint8
                )
                * 255
            )

            mask_patch = (
                mask_canvas[
                    y1:y2,
                    x1:x2,
                ]
            )

            mask_canvas[
                y1:y2,
                x1:x2,
            ] = np.maximum(
                mask_patch,
                hard_mask,
            )

            placed.append(
                (
                    x1,
                    y1,
                    x2,
                    y2,
                )
            )

        self._getitem_calls += 1

        if do_scale_shared:
            self._aug_scale_count += 1

        if applied_geo_any:
            self._aug_geo_count += 1

        if applied_photo_any:
            self._aug_photo_count += 1


        synth_small = cv2.resize(
            canvas[
                pad:pad + 512,
                pad:pad + 512,
            ],
            (
                width,
                height,
            ),
            interpolation=cv2.INTER_AREA,
        )

        mask_512 = mask_canvas[
            pad:pad + 512,
            pad:pad + 512,
        ]

        mask_small = cv2.resize(
            mask_512,
            (
                width,
                height,
            ),
            interpolation=cv2.INTER_NEAREST,
        )

        _, mask_small = (
            cv2.threshold(
                mask_small,
                127,
                255,
                cv2.THRESH_BINARY,
            )
        )
        if self.mode == "anom" and not np.any(mask_small > 0):
            raise RuntimeError(
                "TrainDataset(mode='anom') generated an empty anomaly mask. "
                f"idx={idx}, group={group.name}, "
                f"selected={len(selected)}, placed={len(placed)}"
            )

        if (
            self.save_synthetic
            and self.debug_save_every
            and idx
            % self.debug_save_every
            == 0
        ):
            if (
                self.debug_epoch_only
                is None
                or self.debug_epoch_only
                == 0
            ):
                output_path = (
                    self.dbg_root
                    / (
                        f"synth_e0_"
                        f"{group.name}.png"
                    )
                )

                suffix = 1

                while output_path.exists():
                    output_path = (
                        self.dbg_root
                        / (
                            f"synth_e0_"
                            f"{group.name}_"
                            f"{suffix}.png"
                        )
                    )

                    suffix += 1

                cv2.imwrite(
                    str(output_path),
                    cv2.cvtColor(
                        synth_small,
                        cv2.COLOR_RGB2BGR,
                    ),
                )

        clean_img = self.tr_img(
            cv2.resize(
                background_rgb,
                (
                    width,
                    height,
                ),
                interpolation=cv2.INTER_AREA,
            )
        )

        synth_img = self.tr_img(
            synth_small
        )

        mask_tensor = self.tr_msk(
            mask_small
        )

        mask_exists = torch.tensor(
            1.0
            if np.any(
                mask_small > 0
            )
            else 0.0,
            dtype=torch.float32,
        )

        return (
            clean_img,
            mask_tensor,
            synth_img,
            mask_exists,
        )

    