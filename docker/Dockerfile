# Use PyTorch CUDA base image with Conda
ARG PYTORCH="2.1.0"
ARG CUDA="12.1"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel AS base

# Ensure proper CUDA paths
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX 9.0"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

# Set noninteractive mode to avoid timezone prompt
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies (including OpenCV & Git)
RUN apt-get update && apt-get install -y \
    git \
    libgl1 \
    libglib2.0-0 \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Update Conda and install pip
RUN conda update conda && conda install pip && conda clean -afy

# Set working directory inside the container
WORKDIR /DRAEM

# Copy environment file (if using Conda)
COPY environment.yml /DRAEM/



# Install dependencies via Conda (if applicable)
RUN conda env update -f /DRAEM/environment.yml -n base && conda clean -afy

# Install extra pip packages in the conda base env
RUN /opt/conda/bin/python -m pip install --no-cache-dir debugpy nflows

# Copy project files
COPY . /DRAEM/

# Ensure Python finds installed packages
# ENV PYTHONPATH="/usr/local/lib/python3.10/site-packages:$PYTHONPATH"
ENV PYTHONPATH="/DRAEM:$PYTHONPATH"


# Run the application normally
CMD ["/bin/bash"]










