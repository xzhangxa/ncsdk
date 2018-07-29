FROM ubuntu:16.04

# Install necessary packages for NCSDK
RUN apt-get update && apt-get install -y \
    build-essential \
    libusb-1.0-0-dev \
    pkg-config \
    git \
    cmake \
    python3-pip \
    python3-tk \
    libgraphviz-dev

# Install dependencies of pycaffe
RUN apt-get install -y --no-install-recommends \
    libprotobuf-dev \
    libleveldb-dev \
    libsnappy-dev \
    libopencv-dev \
    libhdf5-serial-dev \
    protobuf-compiler \
    libboost-all-dev \
    libatlas-base-dev \
    python3-dev \
    python3-numpy \
    libgflags-dev \
    libgoogle-glog-dev \
    liblmdb-dev

# Copy over the NCSDK
COPY ./ /ncsdk

# Set the current working directory to the cloned ncsdk directory
WORKDIR "/ncsdk"

ENV PATH "/root/.local/bin:$PATH"

# Build NCSDK and pycaffe
RUN mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release .. && \
    make && \
    make install && \
    make pyncsdk && \
    make pycaffe && \
    pip3 install output/mvnc-*.whl && \
    pip3 install output/mvnctools_lib-*.whl && \
    pip3 install output/mvnctools_compile-*.whl && \
    pip3 install output/mvnctools_check-*.whl && \
    pip3 install output/mvnctools_profile-*.whl

ENV PYTHONPATH "/ncsdk/caffe/python:$PYTHONPATH"

# Regenerate CMake cache to detect pycaffe and build/download/compile apps/data/models.
RUN cd build && cmake .. && make examples

CMD ["/bin/bash"]
