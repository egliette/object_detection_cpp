#!/bin/bash

# Bash "strict mode", to help catch problems and bugs in the shell
# script. Every bash script you write should include this. See
# http://redsymbol.net/articles/unofficial-bash-strict-mode/ for
# details.
set -euo pipefail

# Tell apt-get we're never going to be able to give manual
# feedback:
export DEBIAN_FRONTEND=noninteractive

# Update the package listing, so we know what package exist:
apt-get update

# Install security updates:
apt-get -y upgrade

# Install a new package, without unnecessary recommended packages:

# Install network packages
apt-get -y install --no-install-recommends \
    build-essential \
    iproute2 \
    iputils-ping

# Install Gstreamer
apt-get -y  install --no-install-recommends  \
    htop \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-bad1.0-dev \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    gstreamer1.0-x \
    gstreamer1.0-alsa \
    gstreamer1.0-gl \
    gstreamer1.0-gtk3 \
    gstreamer1.0-qt5 \
    gstreamer1.0-pulseaudio

# Install PyGObject for Gstreamer Python Binding
apt-get -y  install --no-install-recommends \
    pkg-config \
    ffmpeg \
    libcairo2-dev \
    gcc \
    curl \
    python3-dev \
    libgirepository1.0-dev \
    python3-gi \
    gobject-introspection \
    gir1.2-gtk-3.0 \
    graphviz

# Install C tools
apt-get -y  install --no-install-recommends \
    meson \
    ninja-build \
    libopencv-dev \
    wget

# Install ONNXRuntime
wget https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-x64-gpu-1.20.1.tgz
tar -xzf onnxruntime-linux-x64-gpu-1.20.1.tgz
cp -r onnxruntime-linux-x64-gpu-1.20.1/include/* /usr/local/include/
cp -r onnxruntime-linux-x64-gpu-1.20.1/lib/*     /usr/local/lib/
ldconfig

# Delete cached files we don't need anymore (note that if you're
# using official Docker images for Debian or Ubuntu, this happens
# automatically, you don't need to do it yourself):
apt-get clean
# Delete index files we don't need anymore:
rm -rf /var/lib/apt/lists/*
