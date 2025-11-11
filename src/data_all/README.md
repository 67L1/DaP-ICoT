# 📊 Data Preparation for DaP-ICoT

This directory contains the scripts and processed data for the DaP-ICoT project. Follow these instructions to prepare the M3CoT dataset for the main experiment.

### Step 1: 📥 Download the Dataset

First, download the test set for the **M3CoT dataset** from its Hugging Face repository.

- **🤗 Dataset Link:** [M3CoT on Hugging Face](https://huggingface.co/datasets/LightChen2333/M3CoT)

Place the downloaded dataset files into a directory of your choice. You will need to specify the path to these files in the configuration later.

### Step 2: ⚙️ Filter and Convert Dataset

We need to process the raw dataset to filter out entries with missing images and ensure all images are in a consistent format (`.png`).

The `pq_jsonl.py` script handles this process. Run it from the project's root directory (`dap_icot/`).

```bash
# Ensure you are in the root directory 'dap_icot'
python pq_jsonl.py
```

By default, this script will generate a processed `test.jsonl` file and the converted images inside the `data_all/m3cot/` directory.

### Step 3: 🖼️ Generate the Image Pool with SAM2

This step uses the pre-configured **[Segment Anything Model 2 (SAM2)](https://github.com/facebookresearch/sam2)** to perform object detection on the dataset images. The results are stored in an "image pool" file, which is used during the reasoning process.

**a. Move Pre-processing Scripts into the `sam2` directory**

Our custom scripts for this task must be located inside the `sam2` folder to work correctly.

```bash
# Ensure you are in the root directory 'dap_icot'
mv preprocess_pool.py process_res.py sam2_detect.py sam2/
```

**b. Generate the Image Pool**

Now, navigate into the `sam2` directory, configure the paths, and run the pre-processing script.

1.  **Change directory:**
    ```bash
    cd sam2
    ```

2.  **Modify `config.yaml`:**
    Open the `config.yaml` file located inside this `sam2` directory. Adjust the paths to point to your dataset location and the desired output location for the image pool.

3.  **▶️ Run the script:**
    ```bash
    python preprocess_pool.py
    ```

After the script finishes, the generated image pool (e.g., `image_pool_qwen.pkl`) will be stored in the output path you configured, typically `data_all/m3cot/`. 🎉
