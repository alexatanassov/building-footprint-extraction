import os
import numpy as np
import matplotlib.pyplot as plt

LABEL_DIR = "data/tiles/labels"

empty_count = 0
nonempty_count = 0
pixel_coverage = []

label_files = [f for f in os.listdir(LABEL_DIR) if f.endswith('.npy')]

for fname in label_files:
    mask = np.load(os.path.join(LABEL_DIR, fname))
    building_pixels = np.sum(mask == 1)
    total_pixels = mask.size

    if building_pixels == 0:
        empty_count += 1
    else:
        nonempty_count += 1
        pixel_coverage.append(building_pixels / total_pixels)

# Print summary stats
total = empty_count + nonempty_count
print(f"Total label masks: {total}")
print(f"Empty masks (no buildings): {empty_count} ({empty_count / total:.2%})")
print(f"Non-empty masks (with buildings): {nonempty_count} ({nonempty_count / total:.2%})")

# Plot histogram of building pixel coverage (only non-empty masks)
plt.hist(pixel_coverage, bins=20, color='skyblue', edgecolor='black')
plt.xlabel("Building Pixel Coverage (%)")
plt.ylabel("Number of Tiles")
plt.title("Histogram of Building Pixel Coverage in Non-Empty Tiles")
plt.grid(True)
plt.savefig("building_coverage_histogram.png", dpi=200)
plt.show()