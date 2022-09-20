FROM nvcr.io/nvidia/l4t-base:r32.6.1
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        autoconf \
        automake \
        build-essential \
        cmake \
        git \
        libb64-dev \
        libre2-dev \
        libssl-dev \
        libtool \
        libboost-dev \
        libcurl4-openssl-dev \
        libopenblas-dev \
        rapidjson-dev \
        patchelf \
        zlib1g-dev && \
    apt-get autoclean && \
    apt-get autoremove
RUN mkdir -p /opt/triton && \
    wget https://github.com/triton-inference-server/server/releases/download/v2.17.0/tritonserver2.17.0-jetpack4.6.tgz && \
    tar xf tritonserver2.17.0-jetpack4.6.tgz -C /opt/triton && \
    rm tritonserver2.17.0-jetpack4.6.tgz
ENV PATH="/opt/triton/bin:$PATH"
ENV LD_LIBRARY_PATH="/opt/triton/lib:$LD_LIBRARY_PATH"

# download inception model
RUN mkdir -p /tmp/models
RUN mkdir -p /opt/triton/models
# WORKDIR /tmp/models
# COPY . .
# RUN scripts/download_inception.sh
# RUN mv models /opt/triton/models

WORKDIR /opt/triton
ENTRYPOINT ["tritonserver", "--backend-directory=/opt/triton/backends"]