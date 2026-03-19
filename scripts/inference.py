import os
from pathlib import Path
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

# ----------------------------
# Import your preprocessing pipeline
# ----------------------------
from preproc import (
    load_and_normalize,
    segment_fruit,
    refine_mask,
    compute_shape_features,
    compute_colour_features,
    build_feature_vector
)

# ----------------------------
# Config
# ----------------------------
VALID_EXTS = {".jpg", ".jpeg", ".png"}

MODEL_PATH = "catboost_model.cbm"
OUTPUT_CSV = "predictions.csv"


# ----------------------------
# Load trained model
# ----------------------------
model = CatBoostClassifier()
model.load_model(MODEL_PATH)


# ----------------------------
# Feature extraction (single image)
# ----------------------------
def extract_features(image_path):
    img = load_and_normalize(str(image_path))
    raw_mask = segment_fruit(img)
    mask_refined = refine_mask(raw_mask)

    shape_feats = compute_shape_features(mask_refined)
    colour_feats = compute_colour_features(img, mask_refined)

    feature_vector = build_feature_vector(shape_feats, colour_feats)

    return np.array(feature_vector).reshape(1, -1)


# ----------------------------
# Inference over directory
# ----------------------------
def run_inference(input_dir, output_csv=OUTPUT_CSV):
    input_path = Path(input_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Directory not found: {input_dir}")

    rows = []

    image_files = sorted([
        p for p in input_path.rglob("*")
        if p.is_file() and p.suffix.lower() in VALID_EXTS
    ])

    print(f"Found {len(image_files)} images.")

    for img_path in image_files:
        try:
            X = extract_features(img_path)

            pred = model.predict(X)[0]

            # Ensure exact string output
            label = "fruit" if int(pred) == 1 else "vegetable"

            rows.append({
                "image_id": img_path.name,
                "prediction": label
            })

        except Exception as e:
            print(f"Failed on {img_path.name}: {e}")
            rows.append({
                "image_id": img_path.name,
                "prediction": "vegetable"  # fallback (safe default)
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)

    print(f"\nSaved predictions to {output_csv}")
    print(f"Total predictions: {len(df)}")


# ----------------------------
# CLI entry
# ----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str, help="Path to directory of images")
    parser.add_argument("--output", type=str, default=OUTPUT_CSV)

    args = parser.parse_args()

    run_inference(args.input_dir, args.output)