# Base Image from the original Dockerfile (keeping it for compatibility)
FROM nvcr.io/nvidia/deepstream:7.1-triton-multiarch

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Arguments from original Dockerfile
ARG CMAKE_VERSION_MAJOR=3
ARG CMAKE_VERSION_MINOR=25
ARG CMAKE_VERSION_PATCH=3

ARG EIGEN_VERSION_MAJOR=3
ARG EIGEN_VERSION_MINOR=4
ARG EIGEN_VERSION_PATCH=0

ARG OPENCV_VERSION_MAJOR=4
ARG OPENCV_VERSION_MINOR=11
ARG OPENCV_VERSION_PATCH=0

ARG PCL_VERSION_MAJOR=1
ARG PCL_VERSION_MINOR=10
ARG PCL_VERSION_PATCH=0

ARG PYBIND11_VERSION_MAJOR=2
ARG PYBIND11_VERSION_MINOR=13
ARG PYBIND11_VERSION_PATCH=0

ARG YAML_CPP_VERSION_MAJOR=0
ARG YAML_CPP_VERSION_MINOR=8
ARG YAML_CPP_VERSION_PATCH=0

# Install basic dependencies + OpenSSH for RunPod
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libblas-dev \
    libssl-dev \
    liblapack-dev \
    gfortran \
    gnupg \
    software-properties-common \
    libflann-dev \
    libboost-filesystem-dev \
    libboost-date-time-dev \
    libboost-iostreams-dev \
    libboost-system-dev \
    libboost-program-options-dev \
    libzmq3-dev \
    ffmpeg \
    openssh-server \
    curl \
    wget \
    git \
    nano \
    vim \
    htop \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# Setup SSH for RunPod
RUN mkdir -p /var/run/sshd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config

# Install pre-built development libraries (Fast Build)
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    libeigen-dev \
    libopencv-dev \
    libpcl-dev \
    pybind11-dev \
    libyaml-cpp-dev \
    libyaml-cpp-dev \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch
RUN pip3 install -U torch==2.6.0 torchvision==0.21.0

# Modified WORKDIR to avoid conflict with RunPod volume
WORKDIR /app

COPY src/requirements.txt /app/
# Install Python dependencies
RUN pip3 install --no-cache-dir -r /app/requirements.txt

# Install Python dependencies
RUN pip3 install -U pip && pip3 install --no-cache-dir \
    scikit-learn scikit-image --force-reinstall \
    && pip3 install --no-cache-dir --ignore-installed open3d \
    && pip3 install --no-cache-dir kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu124.html \
    && pip3 install --no-cache-dir --no-index pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt241/download.html \
    && pip3 install --no-cache-dir numpy==1.26.4 scipy joblib scikit-learn scikit-image --force-reinstall \
    && pip3 install --no-cache-dir pyrender \
    && pip3 install --no-cache-dir jupyter jupyterlab notebook

# Setup environment
ENV PATH="/bin/python3:${PATH}"
RUN alias python="/bin/python3"
RUN echo 'alias python="/bin/python3"' >> /etc/bash.bashrc && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1 && \
    echo "export PYTHONPATH=/usr/local/lib/python3.10/dist-packages:${PYTHONPATH}" >> /etc/bash.bashrc

RUN cd / && git clone https://github.com/NVLabs/BundleSDF.git

RUN cp -r /BundleSDF/mycuda /customize_cuda
RUN cd /customize_cuda && \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0" FORCE_CUDA=1 pip install . --no-build-isolation
    
# Setup the OpenMPI
RUN mkdir -p /opt/hpcx/ompi/lib/x86_64-linux-gnu
RUN ln -s /opt/hpcx/ompi /opt/hpcx/ompi/lib/x86_64-linux-gnu/openmpi
RUN apt remove -y libvtk9-dev || true

# Reinstall VTK package for OpenCV compatibility
RUN apt-get update && apt-get install -y libvtk9-dev && \
    rm -rf /var/lib/apt/lists/*

# Build and install BundleTrack (copy only necessary artifacts)
RUN cp -r /BundleSDF/BundleTrack /tmp/BundleTrack
RUN cd /tmp/BundleTrack && \
    mkdir -p build && \
    cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && \
    # Copy only the essential built artifacts to standard locations \
    cp my_cpp*.so /usr/local/lib/python3.10/dist-packages/ && \
    cp libBundleTrack.so /usr/local/lib/ && \
    cp libMY_CUDA_LIB.so /usr/local/lib/ && \
    # Update library cache \
    ldconfig && \
    # Remove all source code and build artifacts \
    cd / && \
    rm -rf /tmp/BundleTrack

RUN cd / && \
    git clone https://github.com/NVlabs/FoundationStereo.git 
    
ENV PYTHONPATH=/FoundationStereo/core:$PYTHONPATH

# Install sam2 and roma libraries
# Fix for SAM2: Install in editable mode but keep source
RUN cd / && git clone https://github.com/facebookresearch/sam2.git &&\
    cd sam2 &&\
    SAM2_BUILD_CUDA=0 pip install -e ".[notebooks]" && \ 
    python3 setup.py build_ext --inplace

RUN cd / && git clone https://github.com/Parskatt/RoMa.git &&\
    cd RoMa &&\
    pip3 install . &&\
    cd / && rm -rf /RoMa

# Final cleanup and ldconfig
RUN ldconfig && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /root/.cache/pip && \
    rm -rf /tmp/*

WORKDIR /app

# Copy the entire package structure to /app instead of /workspace
COPY src /app/3d-object-reconstruction/src
COPY README.md /app/3d-object-reconstruction/
COPY notebooks /app/3d-object-reconstruction/notebooks
COPY data /app/3d-object-reconstruction/data

# Install the package in editable mode
WORKDIR /app/3d-object-reconstruction
RUN pip3 install -e src/

# Create Jupyter configuration
RUN jupyter notebook --generate-config && \
    echo "c.NotebookApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.port = 8888" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.open_browser = False" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.allow_root = True" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.token = ''" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.password = ''" >> ~/.jupyter/jupyter_notebook_config.py 
    # Removed c.NotebookApp.allow_origin = '*' for security, but might be needed for RunPod proxy

# Copy start script
COPY start.sh /start.sh
RUN sed -i 's/\r$//' /start.sh && chmod +x /start.sh

# Expose Jupyter port
EXPOSE 8888

# Set the default command to start script
CMD ["/start.sh"]
