import os
import pandas as pd
from PIL import Image
import io
from tqdm import tqdm
import numpy as np

file1 = "/path/to/data/test-00000-of-00002.parquet"
file2 = "/path/t/data/test-00001-of-00002.parquet"

df1 = pd.read_parquet(file1)
df2 = pd.read_parquet(file2)

merge_df = pd.concat([df1, df2], ignore_index=True)

merge_df["choices"] = merge_df["choices"].apply(
    lambda x: [str(i) for i in x] if isinstance(x, (list, np.ndarray)) else x
)

image_dir = "m3cot/images"
os.makedirs(image_dir, exist_ok=True)

for idx, row in tqdm(merge_df.iterrows(), desc="processing image", total=len(merge_df)):
    image_id = str(row["image_id"])
    image_field = row["image"]

    if image_field is None:
        continue

    try:
        if isinstance(image_field, bytes):
            image_data = image_field
        elif isinstance(image_field, dict) and "bytes" in image_field:
            image_data = image_field["bytes"]
        else:
            raise ValueError(f"Unsupported image format for image_id={image_id}")

        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        image.save(os.path.join(image_dir, f"{image_id}.png"))
    except Exception as e:
        print(f"Failed to save image for image_id={image_id}: {e}")

merge_df["image"] = merge_df["image"].apply(lambda x: None if pd.isna(x) or x is None else " ")

merge_df.to_json("m3cot/test.jsonl", orient="records", lines=True, force_ascii=False)
