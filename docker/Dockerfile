FROM nvidia/cuda:10.1-base

# meta data
LABEL maintainer="Fabian Jankowski"
LABEL description="Docker image for heimdall single-pulse pipeline."

RUN apt-get update
RUN apt-get install -y software-properties-common

RUN add-apt-repository -y ppa:kernsuite/kern-5
RUN add-apt-repository -y multiverse
RUN add-apt-repository -y restricted

RUN apt-get update
RUN apt-get install -y \
    dspsr \
    heimdall-astro \
    nvidia-cuda-toolkit \
    psrchive \
    python3 \
    python3-pip \
    sigproc \
    sigpyproc

# add smart history search
RUN echo "\"\e[A\":history-search-backward" >> /etc/skel/.inputrc && \
    echo "\"\e[B\":history-search-forward" >> /etc/skel/.inputrc

# add normal user
ARG USERID=1234
ARG USERNAME=heimdall
RUN adduser --disabled-password --gecos ${USERNAME} \
    --uid ${USERID} ${USERNAME}

# python packages
RUN pip3 install --upgrade pip

RUN pip3 install \
    ipython \
    matplotlib \
    numpy

# install our software in docker image
ARG SOFT_PATH="/software/hdpipe"

RUN mkdir -p ${SOFT_PATH}
COPY . ${SOFT_PATH}
RUN chown -R ${USERID}:${USERID} ${SOFT_PATH}

ENV PYTHONPATH ${SOFT_PATH}:${PYTHONPATH}
WORKDIR ${SOFT_PATH}
