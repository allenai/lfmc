FROM ghcr.io/allenai/pytorch:2.5.1-cuda12.1-python3.11-v2025.03.21

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

WORKDIR /stage

COPY lib/galileo/requirements.txt lib/galileo/requirements.txt
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir data
COPY data/labels data/labels
COPY lib/galileo/config data/config
COPY lib/galileo-data/models data/models

COPY lib/galileo/src galileo
COPY lfmc lfmc

COPY pyproject.toml .
RUN pip install --no-cache-dir -e .
