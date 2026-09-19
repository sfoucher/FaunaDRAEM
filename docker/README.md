# Docker

Container for FaunaDRAEM: CUDA-enabled PyTorch plus the repo's Python dependencies.

## Requirements

- Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (a GPU is needed to *run*, not to build)
- ~10 GB of disk for the image
- Network access on first build (pulls the base image, installs pip packages, clones Segment Anything)

## Build

```bash
./docker/build              # build or rebuild
./docker/build --pull       # refresh the pytorch base image first
./docker/build --no-cache   # rebuild every layer
```

Works from any directory — the script `cd`s to the repository root, which is the build context the
Dockerfile expects. The tag comes from `DOCKER_IMAGE_NAME` in `docker/env` (`draem_image:local`).

The last two layers are smoke tests: they import every third-party dependency, then every
FaunaDRAEM module. A build ending in `FaunaDRAEM imports ok` is a working image — a missing or
ABI-broken dependency fails the build instead of the first training run.

Equivalent without the script:

```bash
docker build -t draem_image:local -f docker/Dockerfile .
```

## Run

```bash
./docker/run                  # interactive shell in /DRAEM
./docker/run python main.py   # run the pipeline directly

DATASETS_DIR=/mnt/data/caribou ./docker/run   # data outside the repo
```

Datasets default to `./DATASETS`, and the script fails with a clear message if that directory does
not exist — Docker would otherwise create the missing path and mount an empty directory. Equivalent
without the script:

```bash
docker run --rm -it --gpus all --ipc=host \
  --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$PWD":/DRAEM \
  -v /path/to/datasets:/DRAEM/DATASETS \
  draem_image:local bash
```

Inside the container:

```bash
python main.py                                    # SAM masks -> train -> test + PDF report
python infer_Draem.py --ckpt ckpt.pt --patches /dir --out /outdir
```

Notes:

- `/DRAEM` is the working directory every absolute path in `main.py` assumes. Keep the mount there.
- The image contains a copy of the code, but the mount shadows it, so host edits are live.
- `--ipc=host` gives DataLoader workers enough shared memory. `--shm-size=8g` is the narrower option.
- The container runs as the calling user, so results written into the mounts are yours, not
  root's. `HOME=/tmp` goes with it: that uid may have no home directory inside the image, and
  matplotlib needs somewhere to put its font cache. The trade-off is that `pip install` inside the
  container fails — start a root container for that:
  `docker run --rm -it --user root -v "$PWD":/DRAEM draem_image:local bash`.
- Datasets and the SAM checkpoint stay on the mounts. `.dockerignore` keeps `DATASETS/`, `*.pt` and
  `*.pth` out of the build context.

## What is in the image

Base: `pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-runtime`, pinned by the `ARG`s at the top
of the Dockerfile. `-runtime`, not `-devel`: nothing here compiles a CUDA extension.

To build against a different pair (Docker Hub publishes torch 2.13.0 with cuda12.6/13.0/13.2, while
cuda12.8 stops at torch 2.11.0):

```bash
docker build --build-arg PYTORCH=2.11.0 --build-arg CUDA=12.8 -t draem_image:local -f docker/Dockerfile .
```

`TORCH_CUDA_ARCH_LIST` is not set in the image; if you reinstate it, sm_120 (Blackwell) needs
CUDA >= 12.8, otherwise kernels load and then fail with `cudaErrorNoKernelImageForDevice`.

Python dependencies are in `docker/requirements.txt`. **torch and torchvision are deliberately absent**
— they come from the base image, and reinstalling them from PyPI would replace the CUDA build.

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| `manifest unknown` on the `FROM` line | `PYTORCH`/`CUDA` are a pair Docker Hub does not publish (see above) |
| `ImportError: libGL.so.1` from `import cv2` | Something pulled in `opencv-python` alongside `opencv-python-headless`. `imgaug` does this, which is why the Dockerfile installs it with `--no-deps` and lists its real dependencies in `requirements.txt` |
| `ModuleNotFoundError: threshold_tunning` | Known: `evaluate.py:21` imports a module that is not in the repo, so `import evaluate` and `python main.py` fail. The Dockerfile's second smoke test leaves `evaluate`/`main` commented out for this reason — uncomment once the module lands |
| numpy errors from `imgaug` | `imgaug` 0.4.0 is unmaintained and breaks on numpy >= 1.24, hence the `numpy<2` pin |
| `could not select device driver` | NVIDIA Container Toolkit missing or not configured on the host |

## Files

| File | Purpose |
| --- | --- |
| `Dockerfile` | Image definition, single `dev` stage |
| `requirements.txt` | Python dependencies (no torch/torchvision) |
| `build` | Build wrapper, sets the context and tag |
| `run` | Container wrapper: mounts, user mapping, `DATASETS_DIR` |
| `env` | Shared image and container names |
