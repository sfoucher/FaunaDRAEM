import random
import warnings
from argparse import ArgumentParser, Namespace
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode as IM
from evaluate import run_test
from generate_masks_sam import generate
from load_sam import download_file
from train import train


warnings.filterwarnings(
    "ignore",
    message=r"Was asked to gather along dimension 0, but all input tensors were scalars",
    category=UserWarning,
    module=r"torch\.nn\.parallel\._functions",
)


SEED_VALUE = 42

random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
torch.manual_seed(SEED_VALUE)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED_VALUE)
    torch.cuda.manual_seed_all(SEED_VALUE)


train_img_tf = T.Compose([
    T.ToPILImage(),
    T.ToTensor(),
    T.Normalize([0.5] * 3, [0.5] * 3),
])

train_msk_tf = T.Compose([
    T.ToPILImage(),
    T.ToTensor(),
])


val_img_tf = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize([0.5] * 3, [0.5] * 3),
])

val_msk_tf = T.Compose([
    T.Resize((256, 256), interpolation=IM.NEAREST),
    T.ToTensor(),
])



def build_args(use_cli: bool = False) -> Namespace:
    """Build experiment configuration."""

    if use_cli:
        parser = ArgumentParser()

        parser.add_argument(
            "--model",
            type=str,
            choices=[
                "draem",
                "padim",
                "padim_nf",
            ],
            default="draem",
        )

        parser.add_argument(
            "--padim_nf_type",
            type=str,
            default="maf",
            choices=[
                "realnvp",
                "maf",
            ],
        )

        parser.add_argument(
            "--padim_nf_heads",
            type=int,
            default=2,
        )

        parser.add_argument(
            "--padim_nf_epochs",
            type=int,
            default=10,
        )
        try:
            from argparse import BooleanOptionalAction

            parser.add_argument(
                "--apply_augmentations",
                action=BooleanOptionalAction,
                default=False,
            )

        except ImportError:
            augmentation_group = parser.add_mutually_exclusive_group()

            augmentation_group.add_argument(
                "--apply_augmentations",
                dest="apply_augmentations",
                action="store_true",
            )

            augmentation_group.add_argument(
                "--no_apply_augmentations",
                dest="apply_augmentations",
                action="store_false",
            )

            parser.set_defaults(
                apply_augmentations=False
            )

        return parser.parse_args()

    return Namespace(
        # Model selection ("draem", "padim", "padim_nf")
        model="draem",
        apply_augmentations=False,
        # Fixed Thresholds
        segmentation_threshold=0.50,
        detection_threshold=0.50,

        lr=1e-4,
        epochs=1,
        batch_size=16,
        num_workers=4,
        det_loss_w=0.1,
        weight_decay=1e-4,
        warmup_epochs=15,

        early_stop_metric="ap",
        monitor_after=10,
        patience=40,
        min_delta=0.001,
        seg_eval_stride=2,

        checkpoint_dir="/DRAEM/Publish_res/checkpoints_New",
        checkpoint_name="draem_final.pt",


        plot_dir="/DRAEM/Publish_res/plots_New",
        seg_plot_dir="/DRAEM/Publish_res/plots_New/segmentation",

        # PaDiM-MAF settings
        
        padim_backbone="wide_resnet50_2",
        padim_num_embeddings=100,
        padim_nf_type="maf",
        padim_nf_heads=1,
        padim_nf_epochs=5,

        # Silhouette bank

        sil_root="/DRAEM/Silhouette_Bank",
        mask_root="/DRAEM/Silhouette_Bank",

        # Training data: empty/normal patches only

        img_root=(
            "/DRAEM/Paper_3_DATASET/Patches_May_31/"
            "Patches_May_31_NOM/AD_Allocation/Train_Empty"
        ),


        csv_file_dir=(
            "/DRAEM/DATASETS/Multi_Herd_data/"
            "Multi_Herd_allocation/Multi_Herd_Patches/TRAIN_gt.csv"
        ),

        # Validation data

        imag_non_empty_dir=(
            "/DRAEM/Paper_3_DATASET/Patches_May_31/"
            "Patches_May_31_NOM/AD_Allocation/val_NOM"
        ),

        masks_path=(
            "/DRAEM/Paper_3_DATASET/Patches_May_31/"
            "Patches_May_31_NOM/AD_Allocation/"
            "Val_test_masks/val"
        ),

        binary_dir=(
            "/DRAEM/May_31_data_RESULTS/"
            "Validation_Results_2/binarymaps"
        ),

        heatmap_dir=(
            "/DRAEM/May_31_data_RESULTS/"
            "Validation_Results_2/heatmaps"
        ),

        anomalymap_dir=(
            "/DRAEM/May_31_data_RESULTS/"
            "Validation_Results_2/anomalyaps"
        ),

        save_val_artifacts=False,

        # Test data

        test2_image_dir=(
            "/DRAEM/Paper_3_DATASET/Patches_May_31/"
            "Patches_May_31_NOM/AD_Allocation/test_NOM"
        ),

        test2_mask_dir=(
            "/DRAEM/Paper_3_DATASET/Patches_May_31/"
            "Patches_May_31_NOM/AD_Allocation/"
            "Val_test_masks/test"
        ),

        # SAM model download and checkpoint path
        url_sam=(
            "https://dl.fbaipublicfiles.com/"
            "segment_anything/sam_vit_h_4b8939.pth"
        ),

        sam_path="/DRAEM/sam_vit_h_4b8939.pth",

        # Transforms

        train_img_tf=train_img_tf,
        train_msk_tf=train_msk_tf,
        val_img_tf=val_img_tf,
        val_msk_tf=val_msk_tf,


        # Evaluation

        eval_scope="hybrid",
    )


def main() -> None:
    """Train DRAEM and evaluate its single final checkpoint."""

    args = build_args(use_cli=False)

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    print(f"[device] {device}")


    download_file(args.url_sam, args.sam_path)
    generate(args)


    print(
        "[thresholds] fixed for entire experiment: "
        f"seg={args.segmentation_threshold:.4f}, "
        f"det={args.detection_threshold:.4f}"
    )

    print("Training started...")


    train_output = train(args)

    checkpoint_path = train_output["best_ckpt"]

    print(
        "[checkpoint] using single final checkpoint: "
        f"{checkpoint_path}"
    )

    print("Starting testing...")

    test2_heatmap_dir = (
        "/DRAEM/Publish_res/heatmaps"
    )

    test2_anomaly_dir = (
        "/DRAEM/Publish_res/anomalymaps"
    )

    test2_binary_dir = (
        "/DRAEM/Publish_res/binarymaps"
    )

    output_pdf_path = (
        "/DRAEM/Publish_res/test_metrics_2.pdf"
    )

    test2_metrics = run_test(
        ckpt_path=checkpoint_path,
        img_dir=args.test2_image_dir,
        mask_dir=args.test2_mask_dir,
        batch=8,
        device=device,
        output_pdf_path=output_pdf_path,
        heatmap_dir=test2_heatmap_dir,
        anomaly_map_dir=test2_anomaly_dir,
        binary_dir=test2_binary_dir,
        args=args,
    )

    print("\n===== TEST-2 results =====")

    print("─ Segmentation (pixel-level, global hybrid) ─")

    for label, key in [
        (
            "f1",
            "seg_f1_global",
        ),
        (
            "precision",
            "seg_precision_global",
        ),
        (
            "recall",
            "seg_recall_global",
        ),
        (
            "ap",
            "seg_ap_global",
        ),
        (
            "auroc",
            "seg_auc_global",
        ),
    ]:
        metric_value = test2_metrics.get(
            key,
            float("nan"),
        )

        print(
            f"  {label:<10}: "
            f"{metric_value:.4f}"
        )


    print("\n─ Detection (patch-level) ─")

    for label, key in [
        (
            "f1",
            "det_f1",
        ),
        (
            "precision",
            "det_precision",
        ),
        (
            "recall",
            "det_recall",
        ),
        (
            "ap",
            "det_ap",
        ),
        (
            "auroc",
            "det_auroc",
        ),
    ]:
        metric_value = test2_metrics.get(
            key,
            float("nan"),
        )

        print(
            f"  {label:<10}: "
            f"{metric_value:.4f}"
        )

if __name__ == "__main__":
        main()    

    

    