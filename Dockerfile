FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y python3 python3-pip python3-venv && rm -rf /var/lib/apt/lists/*
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN python -m venv /venv
# Make sure we use the virtualenv:
ENV PATH="/venv/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV TERM=xterm-256color
RUN sed -i 's/#force_color_prompt=yes/force_color_prompt=yes/g' /root/.bashrc
RUN python -m pip install --upgrade pip

COPY install-packages.sh .
RUN ./install-packages.sh

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install opencv-python-headless
