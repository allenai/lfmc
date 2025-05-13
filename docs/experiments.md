# Experiments

This page lists the experiments that were run for the paper.

## Setup

Define the following environment variables in a .env file:

```
BEAKER_BUDGET=...
BEAKER_WORKSPACE=...
BEAKER_CLUSTERS=...
WEKA_BUCKET=...
```

## Default

### Pretrained weights

```shell
python -m experiment.beaker_finetune \
    --image-name ${USER}/lfmc \
    --priority high \
    --gpu-count 1 \
    --model-name tiny \
    --data-folder /presto/data/lfmc/training_tifs \
    --h5py-folder /presto/data/lfmc/h5pys \
    --h5pys-only
```

### Random weights

```shell
python -m experiment.beaker_finetune \
    --image-name ${USER}/lfmc \
    --priority high \
    --gpu-count 1 \
    --model-name tiny \
    --data-folder /presto/data/lfmc/training_tifs \
    --h5py-folder /presto/data/lfmc/h5pys \
    --h5pys-only \
    --no-load-weights
```

## Data ablations

### All minus Sentinel-1

#### Pretrained weights

```shell
python -m experiment.beaker_finetune \
    --image-name ${USER}/lfmc \
    --priority high \
    --gpu-count 1 \
    --model-name tiny \
    --data-folder /presto/data/lfmc/training_tifs \
    --h5py-folder /presto/data/lfmc/h5pys \
    --h5pys-only \
    --excluded-bands S1
```

#### Random weights

```shell
python -m experiment.beaker_finetune \
    --image-name ${USER}/lfmc \
    --priority high \
    --gpu-count 1 \
    --model-name tiny \
    --data-folder /presto/data/lfmc/training_tifs \
    --h5py-folder /presto/data/lfmc/h5pys \
    --h5pys-only \
    --excluded-bands S1 \
    --no-load-weights
```

### All minus Sentinel-2

#### Pretrained weights

```shell
python -m experiment.beaker_finetune \
    --image-name ${USER}/lfmc \
    --priority high \
    --gpu-count 1 \
    --model-name tiny \
    --data-folder /presto/data/lfmc/training_tifs \
    --h5py-folder /presto/data/lfmc/h5pys \
    --h5pys-only \
    --excluded-bands S2_RGB S2_Red_Edge S2_NIR_10m S2_NIR_20m S2_SWIR NDVI
```

#### Random weights

```shell
python -m experiment.beaker_finetune \
    --image-name ${USER}/lfmc \
    --priority high \
    --gpu-count 1 \
    --model-name tiny \
    --data-folder /presto/data/lfmc/training_tifs \
    --h5py-folder /presto/data/lfmc/h5pys \
    --h5pys-only \
    --excluded-bands S2_RGB S2_Red_Edge S2_NIR_10m S2_NIR_20m S2_SWIR NDVI \
    --no-load-weights
```

### All minus ERA5

#### Pretrained weights

```shell
python -m experiment.beaker_finetune \
    --image-name ${USER}/lfmc \
    --priority high \
    --gpu-count 1 \
    --model-name tiny \
    --data-folder /presto/data/lfmc/training_tifs \
    --h5py-folder /presto/data/lfmc/h5pys \
    --h5pys-only \
    --excluded-bands ERA5
```

#### Random weights

```shell
python -m experiment.beaker_finetune \
    --image-name ${USER}/lfmc \
    --priority high \
    --gpu-count 1 \
    --model-name tiny \
    --data-folder /presto/data/lfmc/training_tifs \
    --h5py-folder /presto/data/lfmc/h5pys \
    --h5pys-only \
    --excluded-bands ERA5 \
    --no-load-weights
```

### All minus TerraClimate

#### Pretrained weights

```shell
python -m experiment.beaker_finetune \
    --image-name ${USER}/lfmc \
    --priority high \
    --gpu-count 1 \
    --model-name tiny \
    --data-folder /presto/data/lfmc/training_tifs \
    --h5py-folder /presto/data/lfmc/h5pys \
    --h5pys-only \
    --excluded-bands TC
```

#### Random weights

```shell
python -m experiment.beaker_finetune \
    --image-name ${USER}/lfmc \
    --priority high \
    --gpu-count 1 \
    --model-name tiny \
    --data-folder /presto/data/lfmc/training_tifs \
    --h5py-folder /presto/data/lfmc/h5pys \
    --h5pys-only \
    --excluded-bands TC \
    --no-load-weights
```

### All minus SRTM

#### Pretrained weights

```shell
python -m experiment.beaker_finetune \
    --image-name ${USER}/lfmc \
    --priority high \
    --gpu-count 1 \
    --model-name tiny \
    --data-folder /presto/data/lfmc/training_tifs \
    --h5py-folder /presto/data/lfmc/h5pys \
    --h5pys-only \
    --excluded-bands SRTM
```

#### Random weights

```shell
python -m experiment.beaker_finetune \
    --image-name ${USER}/lfmc \
    --priority high \
    --gpu-count 1 \
    --model-name tiny \
    --data-folder /presto/data/lfmc/training_tifs \
    --h5py-folder /presto/data/lfmc/h5pys \
    --h5pys-only \
    --excluded-bands SRTM \
    --no-load-weights
```

### All minus location

#### Pretrained weights

```shell
python -m experiment.beaker_finetune \
    --image-name ${USER}/lfmc \
    --priority high \
    --gpu-count 1 \
    --model-name tiny \
    --data-folder /presto/data/lfmc/training_tifs \
    --h5py-folder /presto/data/lfmc/h5pys \
    --h5pys-only \
    --excluded-bands location
```

#### Random weights

```shell
python -m experiment.beaker_finetune \
    --image-name ${USER}/lfmc \
    --priority high \
    --gpu-count 1 \
    --model-name tiny \
    --data-folder /presto/data/lfmc/training_tifs \
    --h5py-folder /presto/data/lfmc/h5pys \
    --h5pys-only \
    --excluded-bands location \
    --no-load-weights
```

## Shape ablations

### output_hw=16, num_timesteps=12

#### Pretrained weights

```shell
python -m experiment.beaker_finetune \
    --image-name ${USER}/lfmc \
    --priority high \
    --gpu-count 1 \
    --model-name tiny \
    --data-folder /presto/data/lfmc/training_tifs \
    --h5py-folder /presto/data/lfmc/h5pys \
    --h5pys-only \
    --output-hw 16 \
    --num-timesteps 12
```

#### Random weights

```shell
python -m experiment.beaker_finetune \
    --image-name ${USER}/lfmc \
    --priority high \
    --gpu-count 1 \
    --model-name tiny \
    --data-folder /presto/data/lfmc/training_tifs \
    --h5py-folder /presto/data/lfmc/h5pys \
    --h5pys-only \
    --output-hw 16 \
    --num-timesteps 12 \
    --no-load-weights
```

### output_hw=32, num_timesteps=6

#### Pretrained weights

```shell
python -m experiment.beaker_finetune \
    --image-name ${USER}/lfmc \
    --priority high \
    --gpu-count 1 \
    --model-name tiny \
    --data-folder /presto/data/lfmc/training_tifs \
    --h5py-folder /presto/data/lfmc/h5pys \
    --h5pys-only \
    --output-hw 32 \
    --num-timesteps 6
```

#### Random weights

```shell
python -m experiment.beaker_finetune \
    --image-name ${USER}/lfmc \
    --priority high \
    --gpu-count 1 \
    --model-name tiny \
    --data-folder /presto/data/lfmc/training_tifs \
    --h5py-folder /presto/data/lfmc/h5pys \
    --h5pys-only \
    --output-hw 32 \
    --num-timesteps 6 \
    --no-load-weights
```

### output_hw=32, num_timesteps=3

#### Pretrained weights

```shell
python -m experiment.beaker_finetune \
    --image-name ${USER}/lfmc \
    --priority high \
    --gpu-count 1 \
    --model-name tiny \
    --data-folder /presto/data/lfmc/training_tifs \
    --h5py-folder /presto/data/lfmc/h5pys \
    --h5pys-only \
    --output-hw 32 \
    --num-timesteps 3
```

#### Random weights

```shell
python -m experiment.beaker_finetune \
    --image-name ${USER}/lfmc \
    --priority high \
    --gpu-count 1 \
    --model-name tiny \
    --data-folder /presto/data/lfmc/training_tifs \
    --h5py-folder /presto/data/lfmc/h5pys \
    --h5pys-only \
    --output-hw 32 \
    --num-timesteps 3 \
    --no-load-weights
```

### output_hw=1, patch_size=1, num_timesteps=12

#### Pretrained weights

```shell
python -m experiment.beaker_finetune \
    --image-name ${USER}/lfmc \
    --priority high \
    --gpu-count 1 \
    --model-name tiny \
    --data-folder /presto/data/lfmc/training_tifs \
    --h5py-folder /presto/data/lfmc/h5pys \
    --h5pys-only \
    --output-hw 1 \
    --patch-size 1 \
    --num-timesteps 12
```

#### Random weights

```shell
python -m experiment.beaker_finetune \
    --image-name ${USER}/lfmc \
    --priority high \
    --gpu-count 1 \
    --model-name tiny \
    --data-folder /presto/data/lfmc/training_tifs \
    --h5py-folder /presto/data/lfmc/h5pys \
    --h5pys-only \
    --output-hw 1 \
    --patch-size 1 \
    --num-timesteps 12 \
    --no-load-weights
```

### output_hw=8, patch_size=8, num_timesteps=12

#### Pretrained weights

```shell
python -m experiment.beaker_finetune \
    --image-name ${USER}/lfmc \
    --priority high \
    --gpu-count 1 \
    --model-name tiny \
    --data-folder /presto/data/lfmc/training_tifs \
    --h5py-folder /presto/data/lfmc/h5pys \
    --h5pys-only \
    --output-hw 8 \
    --patch-size 8 \
    --num-timesteps 12
```

#### Random weights

```shell
python -m experiment.beaker_finetune \
    --image-name ${USER}/lfmc \
    --priority high \
    --gpu-count 1 \
    --model-name tiny \
    --data-folder /presto/data/lfmc/training_tifs \
    --h5py-folder /presto/data/lfmc/h5pys \
    --h5pys-only \
    --output-hw 8 \
    --patch-size 8 \
    --num-timesteps 12 \
    --no-load-weights
```

### Spatial splits

#### Pretrained weights

```shell
python -m experiment.beaker_finetune \
    --image-name ${USER}/lfmc \
    --priority high \
    --gpu-count 1 \
    --model-name tiny \
    --data-folder /presto/data/lfmc/training_tifs \
    --h5py-folder /presto/data/lfmc/h5pys \
    --h5pys-only \
    --validation-state-regions Colorado "New Mexico" \
    --test-state-regions Texas Michigan
```

#### Pretrained weights

```shell
python -m experiment.beaker_finetune \
    --image-name ${USER}/lfmc \
    --priority high \
    --gpu-count 1 \
    --model-name tiny \
    --data-folder /presto/data/lfmc/training_tifs \
    --h5py-folder /presto/data/lfmc/h5pys \
    --h5pys-only \
    --validation-state-regions Colorado "New Mexico" \
    --test-state-regions Texas Michigan \
    --no-load-weights
```
