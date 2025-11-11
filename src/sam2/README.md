# ⚙️ SAM2 Setup for DaP-ICoT

This directory contains the **[Segment Anything Model 2 (SAM2)](https://github.com/facebookresearch/sam2)** repository, which is used for object detection during the data pre-processing stage of the DaP-ICoT project.

Follow the steps below to set up SAM2 correctly. 🤖

### 1. 🐑 Clone the SAM2 Repository

First, you need to clone the official SAM2 repository into this directory.

```bash
# This command should be run from the project root (dap_icot/)
# The result will be a 'sam2' folder containing the repository content.
git clone https://github.com/facebookresearch/sam2.git
```

> **📝 Note**: If you are already inside this `sam2` directory, it means the repository has likely been cloned. You can proceed to the next step.

### 2. 💾 Download Model Checkpoints

After cloning the repository, you need to download the pre-trained SAM2 model weights. The official repository provides a script for this.

```bash
# Navigate into the sam2 directory if you are not already there
cd sam2

# Create the checkpoints directory and download the weights
mkdir -p checkpoints && cd checkpoints

# On some systems, you might need to make the script executable first:
# chmod +x download_ckpts.sh
./download_ckpts.sh

# Navigate back to the sam2 directory
cd ../
```

After running the script, the necessary model checkpoints will be located in the `sam2/checkpoints/` directory.

✅ Once these steps are complete, SAM2 is ready to be used for data pre-processing as described in the `data_all/README.md` file.
