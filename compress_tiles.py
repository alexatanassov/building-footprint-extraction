import os
import numpy as np
import argparse
from tqdm import tqdm

# === CONFIGURATION ===
IMG_DIR = "data/tiles/images"
LABEL_DIR = "data/tiles/labels"
IMG_OUT_DIR = "data/tiles_npz/images"
LABEL_OUT_DIR = "data/tiles_npz/labels"
DELETE_ORIGINALS = False  # Set to True if you want to delete .npy files after compression

# === Ensure output folders exist ===
os.makedirs(IMG_OUT_DIR, exist_ok=True)
os.makedirs(LABEL_OUT_DIR, exist_ok=True)

def compress_npy_to_npz(src_dir, dst_dir, delete_original=False):
    files = [f for f in os.listdir(src_dir) if f.endswith(".npy")]
    for fname in tqdm(files, desc=f"Compressing {os.path.basename(src_dir)}"):
        path = os.path.join(src_dir, fname)
        arr = np.load(path)
        out_path = os.path.join(dst_dir, fname.replace(".npy", ".npz"))
        np.savez_compressed(out_path, arr)
        if delete_original:
            os.remove(path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--delete", action="store_true", help="Delete original .npy files after compression")
    args = parser.parse_args()

    compress_npy_to_npz(IMG_DIR, IMG_OUT_DIR, delete_original=args.delete)
    compress_npy_to_npz(LABEL_DIR, LABEL_OUT_DIR, delete_original=args.delete)

if __name__ == "__main__":
    main()