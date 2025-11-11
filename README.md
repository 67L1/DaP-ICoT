# DaP-ICoT: Let's Think with Images Efficiently!
### Official PyTorch Implementation for "Let's Think with Images Efficiently! An Interleaved-Modal Chain-of-Thought Reasoning Framework with Dynamic and Precise Visual Thoughts"

---

## 📖 Introduction

This repository contains the official implementation for our paper, **DaP-ICoT**.

Recently, Interleaved-modal Chain-of-Thought (ICoT) reasoning has shown promising performance by leveraging both multimodal inputs and outputs. However, existing ICoT methods suffer from two fundamental limitations:
1.  **Static Visual Thought Positioning**: Visual information is statically inserted at fixed steps, leading to inefficient and inflexible reasoning.
2.  **Broken Visual Thought Representation**: Fragmented visual cues hinder semantic coherence and precision, undermining the quality of the reasoning process.

To address these critical issues, we introduce **DaP-ICoT**, an Interleaved-modal Chain-of-Thought reasoning framework with **D**ynamic **a**nd **P**recise Visual Thoughts.

## ✨ Key Features

DaP-ICoT incorporates two key components to revolutionize ICoT reasoning:

*   🧠 **Dynamic Visual Thought Integration**: Adaptively introduces visual inputs based on the model's real-time reasoning needs. This reduces redundancy by focusing only on key visual cues, making the process more efficient and human-like.

*   🎯 **Precise Visual Thought Guidance**: Ensures that the generated visual representations are semantically coherent and contextually aligned with the reasoning chain. This enhances the accuracy and reliability of the model's outputs.

Our experiments across multiple benchmarks and models demonstrate that DaP-ICoT not only achieves **state-of-the-art performance** but also significantly improves efficiency. It leads to a **72.6% decrease in token consumption** by reducing the number of inserted images, paving the way for more practical and scalable ICoT reasoning.

---

## 📋 Table of Contents

- [Introduction](#-introduction)
- [Key Features](#-key-features)
- [Prerequisites](#️-prerequisites)
- [Installation and Setup](#-installation-and-setup)
- [Data Preparation](#-data-preparation)
- [Running the Code](#️-running-the-code)
- [Project Structure](#-project-structure)

---

## ⚙️ Prerequisites

- Python 3.10
- Conda
- Git

---

## 🚀 Installation and Setup

Follow these steps carefully to set up the project environment and all necessary components.

‼️ Note: All the following operations are performed in the src directory.

### 1. Clone the Repository

First, clone this repository to your local machine.

```bash 
git clone https://github.com/67L1/DaP-ICoT.git
cd dap_icot
```

### 2. Create Conda Environment

We recommend using Conda to manage dependencies. Create and activate a new environment with Python 3.10.

```bash
conda create -n dapicot python=3.10
conda activate dapicot
```

### 3. Install Python Dependencies

Install all the required Python packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```


> 💡 **Note on PyTorch Installation**
>
> The `requirements.txt` file may not automatically install the correct version of PyTorch for your specific hardware (especially CUDA). If you encounter errors related to `torch` or CUDA during the installation, we **strongly recommend** installing PyTorch manually first.
>
> **For CUDA 12.1**, you can use the following command:
> ```bash
> pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
> ```
>
> For other CUDA versions or CPU-only installations, please visit the official PyTorch website to find the correct command for your system. This will ensure full compatibility.
> - **Link:** [PyTorch Previous Versions Page](https://pytorch.org/get-started/previous-versions/)

### 4. Patch the `transformers` Library

> ⚠️ **IMPORTANT**: This project requires a manual modification to the `transformers` library to support custom visual token handling for the Qwen model. Without this patch, the model will not function correctly.

You need to find the `utils.py` file within your installed `transformers` library and modify the `_sample` method of the `GenerationMixin` class.

**a. Find the file location:**
You can find the path to `utils.py` by running this Python command in your activated `dapicot` environment:

```bash
python -c "import transformers; import os; print(os.path.join(os.path.dirname(transformers.__file__), 'generation', 'utils.py'))"
```
This will print the full path to the file you need to edit.

**b. Apply the patch:**
Open the `utils.py` file and locate the following line (around line 3257):

```python
# update generated ids, model inputs, and length for next step
input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
```

**Replace** this line with the code block below:

```diff
-            # update generated ids, model inputs, and length for next step
-            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

+            # update generated ids, model inputs, and length for next step
+
+            # qwen
+            if 'selected_vokens' in outputs and outputs['selected_vokens'] is not None:
+                # if outputs['selected_vokens'].shape[0] != 1 :
+                num_vokens = outputs['selected_vokens'].shape[0]
+                voken_ids = torch.full(
+                    (1, num_vokens),
+                    fill_value=151655,
+                    dtype=input_ids.dtype,
+                    device=input_ids.device
+                )
+                start_token = torch.full((1, 1), 151652, dtype=input_ids.dtype, device=input_ids.device)
+                end_token = torch.full((1, 1), 151653, dtype=input_ids.dtype, device=input_ids.device)
+                input_ids = torch.cat([input_ids, start_token, voken_ids, end_token, next_tokens[:, None]], dim=-1)
+            else:
+                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
```

### 5. Setup SAM2

We use [Segment Anything Model 2 (SAM2)](https://github.com/facebookresearch/sam2) for object detection.

> 💡 **Note on SAM2 Dependencies**
>
> SAM2 has its own set of dependencies. Although our `requirements.txt` covers all of them, if you encounter any installation or dependency errors specifically when running SAM2 scripts, please refer to the official [SAM2 GitHub repository](https://github.com/facebookresearch/sam2) for detailed installation instructions and troubleshooting.

**a. Clone the SAM2 repository:**

```bash
git clone https://github.com/facebookresearch/sam2.git
```

**b. Download SAM2 checkpoints:**
Navigate into the `sam2` directory and run the official script to download the model weights.

```bash
cd sam2/checkpoints
# On some systems you might need to make the script executable first: chmod +x download_ckpts.sh
./download_ckpts.sh
cd ../../
```
This will place the checkpoints in the `src/sam2/checkpoints/` directory.

💡 We use [sam2.1_hiera_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt) as our tool.

---

## 📊 Data Preparation

### 1. Download the M3CoT Dataset

Download the test set for the M3CoT dataset from Hugging Face:
- **Dataset Link:** [M3CoT](https://huggingface.co/datasets/LightChen2333/M3CoT)

Place the downloaded files into a directory of your choice. You will need to specify this path later in the config file.

### 2. Filter and Convert Dataset

Run the `pq_jsonl.py` script to filter out entries with empty images and convert all images to the `.png` format. This script will generate a `test.jsonl` file.

```bash
cd data_all
python pq_jsonl.py
```
By default, the output `test.jsonl` and processed images will be stored in the `data_all/m3cot/` directory.

### 3. Prepare SAM2 for Pre-processing

**a. Move custom scripts into the `sam2` directory:**
Our custom scripts for SAM2 pre-processing must be located inside the `sam2` folder.

```bash
# Ensure you are in the root directory 'DaP_ICoT/src'
mv preprocess_pool.py process_res.py sam2_detect.py sam2/
```

**b. Generate the image pool:**
This step uses SAM2 to detect objects in the dataset images and creates a pre-processed "image pool".

```bash
cd sam2
```
Next, **modify the `config.yaml` file** located in the `src/config/` directory. You will need to set the `sam2_checkpoint` path and the correct path for your dataset.

💡 **We recommend using absolute paths directly for these settings.**

After configuring, run the script:
```bash
python preprocess_pool.py
```
The resulting image pool will be stored in `data_all/m3cot/` (or your configured path).

---

## ▶️ Running the Code

Now you are ready to run the main experiment.

1.  **Navigate back to the project root directory `DaP-ICoT/src`:**

    ```bash
    cd ../  # If you are still in the sam2 directory
    ```

2.  **Configure the main run:**
    Before running, open the main `config.yaml` file in the project's root directory (`dap_icot/config.yaml`). **Adjust the paths** and other parameters as needed for your setup.

3.  **Execute the main script:**

    ```bash
    python run.py
    ```

---


## 📂 Project Structure

Here is a simplified overview of the project directory structure:

```
dap_icot/src/
├── sam2/                        # Cloned SAM2 repository
│   ├── checkpoints/
│   ├── config.yaml              # Config for SAM2 pre-processing
│   ├── preprocess_pool.py       # (Moved here)
│   └── ...
├── data_all/
│   ├── pq_jsonl.py              # Dataset filtering script
│   └── m3cot/                   # Processed data and image pools
|       └── images/              # M3CoT's images
│       └── test.jsonl            
│       └── image_pool_qwen.pkl  # Image pool for Qwen
├── config/                      # Main configuration file for run.py
│   └── config.yaml              
├── requirements.txt             # Python dependencies
├── run.py                       # Main script to run the experiment
└── README.md
```
