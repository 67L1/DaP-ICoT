# Image Pools for DaP-ICoT

This directory is intended to store the pre-processed **image pools** required to run the DaP-ICoT model.

## What is an Image Pool?

An image pool is a file (e.g., a `.pkl` file) that contains detected objects and visual features extracted from the dataset images using the Segment Anything Model 2 (SAM2).

These pools are essential for the **Dynamic and Precise Visual Thought** mechanism, as they provide the main model with a ready-to-use, structured set of visual cues to incorporate into its reasoning chain.

## How Are These Files Generated?

The files in this directory are **not downloaded**. You must generate them by following the data pre-processing steps outlined in the project.

For detailed instructions on how to create an image pool, please refer to **Step 3: Generate the Image Pool with SAM2** in the `data_all/README.md` file.

## Expected Directory Structure

The image pools should be organized by dataset. After running the pre-processing scripts, your directory structure should look similar to this:

```
image_pools/
└── m3cot/
    └── image_pool_qwen.pkl
```

## Usage

The main experiment script (`run.py`) loads these image pool files during execution.

> ⚠️ **IMPORTANT**: Before running the main experiment, please ensure that the path to your generated image pool file (e.g., `image_pools/m3cot/image_pool_qwen.pkl`) is correctly specified in the main `config.yaml` file located in the project's root directory.
