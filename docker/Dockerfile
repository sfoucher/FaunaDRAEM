# FaunaDRAEM container: CUDA-enabled PyTorch + the repo's Python dependencies.
#
#   Build:  ./docker/build                  (from anywhere in the repo)
#   Run:    docker run --rm -it --gpus all --ipc=host \
#             -v "$PWD":/DRAEM -v /path/to/datasets:/DRAEM/DATASETS draem_image:local bash
#
# The image already contains a copy of the code, but the usual workflow bind-mounts the repo over
# /DRAEM so host edits are live. Data and the SAM checkpoint stay on the mounts, never in the image.
# Everything below assumes the container working dir /DRAEM, which is what main.py's absolute paths
# expect. Requires the NVIDIA Container Toolkit on the host; ~10 GB of disk for the image.
#
# The three ARGs below are overridable at build time, e.g. to stay on CUDA 12.8:
#   docker build --build-arg PYTORCH=2.11.0 --build-arg CUDA=12.8 -f docker/Dockerfile .
# Arch list must cover the deploy GPUs; sm_120 (Blackwell) needs CUDA >= 12.8, or the image will
# load fine and then fail every kernel launch with cudaErrorNoKernelImageForDevice.
# Valid pairs on Docker Hub: 2.13.0 ships cuda12.6/13.0/13.2, cuda12.8 stops at torch 2.11.0.
ARG PYTORCH="2.13.0"
ARG CUDA="13.0"
ARG CUDNN="9"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-runtime AS dev

# The image ships Ubuntu's python3.12, which is PEP 668 externally-managed. A container is already
# an isolated environment, so install into it directly rather than nesting a venv (a venv would also
# need --system-site-packages to see the prebuilt torch in /usr/local/lib/python3.12).
# Set as env rather than a flag so the editable reinstall in docker/run inherits it too.
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PYTHONPATH=/DRAEM

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /DRAEM

# torch/torchvision come from the base image; reinstalling them would clobber the CUDA build.
COPY docker/requirements.txt .
# imgaug is installed --no-deps on purpose: it requires the GL-linked opencv-python, which would
# shadow opencv-python-headless and make `import cv2` fail on libGL.so.1. Its real deps are in
# requirements.txt instead.
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --no-deps imgaug

COPY . /DRAEM/

# Smoke test: fail the build, not the first training run, if a dep is missing or ABI-broken.
# CUDA itself is untestable here (no GPU at build time), so only the build flavour is printed.
RUN python -c "import numpy, torch, torchvision, cv2, imgaug, skimage, sklearn, scipy, pandas, \
matplotlib, tqdm, requests, nflows, segment_anything, PIL; \
print('deps ok | torch', torch.__version__, '| cuda', torch.version.cuda, '| numpy', numpy.__version__)"

# Smoke test 2: the repo's own modules must import against those deps.
# Extract_silhouet_boxes_from_masks is excluded on purpose - it has no __main__ guard and
# runs extraction on absolute host paths at import time.
RUN python -c "import model, loss, metrics, utils, train, train_dataset, test_dataset, \
generate_masks_sam, load_sam, infer_Draem, post_processing, threshold_tunning, evaluate, main, \
Padim.MAF, Padim.padim_adapter, Padim.padim_nf_adapter; \
print('FaunaDRAEM imports ok')"

# No entrypoint script: the container drops into a shell. Inside it, `python main.py` runs the full
# SAM -> train -> evaluate pipeline (edit build_args() in main.py to point at your data first).
CMD ["/bin/bash"]
