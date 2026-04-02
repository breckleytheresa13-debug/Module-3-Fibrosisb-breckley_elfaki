#%%

# ---------------------------
# Import Libraries
# ---------------------------
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ---------------------------
# Load CSV with filenames + depths
# ---------------------------
csv_path = r"C:\Users\15712\OneDrive - University of Virginia\Comp Mod 3\Module-3-Fibrosisb-breckley_elfaki\Filenames and Depths for Students.csv"
df_files = pd.read_csv(csv_path)

print("CSV columns:", df_files.columns.tolist())

filename_col = df_files.columns[0]
depth_col = df_files.columns[1]

# ---------------------------
# Full path to image folder
# ---------------------------
image_folder = r"C:\Users\15712\OneDrive - University of Virginia\Comp Mod 3\Module-3-Fibrosisb-breckley_elfaki\images"

# ---------------------------
# Function to calculate pixel data
# ---------------------------
def calculate_fibrosis(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Warning: Could not load {image_path}")
        return None, None, None

    # Binary threshold
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Pixel counts
    white_pixels = np.sum(binary == 255)
    total_pixels = binary.size
    black_pixels = total_pixels - white_pixels

    # Percentage
    white_percent = (white_pixels / total_pixels) * 100

    return white_pixels, black_pixels, white_percent

# ---------------------------
# Process all images
# ---------------------------
filenames_used = []
depths = []
white_values = []
black_values = []
percent_values = []

for _, row in df_files.iterrows():
    filename = str(row[filename_col]).strip()

    # Remove extra folder text if present
    filename = filename.replace("images/", "").replace("images\\", "")

    depth = row[depth_col]

    image_path = os.path.join(image_folder, filename)

    if not os.path.exists(image_path):
        print("Missing file:", image_path)
        continue

    white, black, percent = calculate_fibrosis(image_path)

    if white is not None:
        filenames_used.append(filename)
        depths.append(depth)
        white_values.append(white)
        black_values.append(black)
        percent_values.append(percent)

# ---------------------------
# Create DataFrame
# ---------------------------
results_df = pd.DataFrame({
    "Filename": filenames_used,
    "Depth (microns)": depths,
    "White pixels": white_values,
    "Black pixels": black_values,
    "White percent": percent_values
})

# Convert depth to numeric and sort
results_df["Depth (microns)"] = pd.to_numeric(results_df["Depth (microns)"], errors="coerce")
results_df = results_df.dropna()
results_df = results_df.sort_values(by="Depth (microns)")

# ---------------------------
# Save CSV file
# ---------------------------
output_csv = r"C:\Users\15712\OneDrive - University of Virginia\Comp Mod 3\Module-3-Fibrosisb-breckley_elfaki\fibrosis_vs_depth_results.csv"
results_df.to_csv(output_csv, index=False)

print(f"\nCSV saved as: {output_csv}")
print(results_df.head())

# ---------------------------
# Verification check
# ---------------------------
total_pixels_check = np.array(white_values) + np.array(black_values)
print("Pixel count check passed:", np.all(total_pixels_check > 0))

# ---------------------------
# Plot RAW data
# ---------------------------
x = results_df["Depth (microns)"].values
y = results_df["White percent"].values

plt.figure(figsize=(8, 5))
plt.scatter(x, y, alpha=0.6)
plt.xlabel("Depth (microns)")
plt.ylabel("Fibrosis (%)")
plt.title("Fibrosis vs Depth")
plt.grid(True)
plt.show()

# ---------------------------
# Moving Average Trend Line
# ---------------------------
# Sort data (just to be safe)
sort_idx = np.argsort(x)
x_sorted = x[sort_idx]
y_sorted = y[sort_idx]

# Moving average
window = 7
y_smooth = np.convolve(y_sorted, np.ones(window)/window, mode='valid')
x_smooth = x_sorted[window-1:]

plt.figure(figsize=(8, 5))
plt.scatter(x_sorted, y_sorted, alpha=0.4, label="Raw data")
plt.plot(x_smooth, y_smooth, linewidth=2, label="Moving average trend")
plt.xlabel("Depth (microns)")
plt.ylabel("Fibrosis (%)")
plt.title("Fibrosis vs Depth with Smoothed Trend")
plt.grid(True)
plt.legend()
plt.show()

#%%
# ---------------------------
# Zoomed-in regions
# ---------------------------
def plot_zoom(x, y, xmin, xmax, title):
    mask = (x >= xmin) & (x <= xmax)
    
    x_zoom = x[mask]
    y_zoom = y[mask]
    
    if len(x_zoom) == 0:
        print(f"No data in range {xmin}-{xmax}")
        return
    
    
    plt.figure(figsize=(7,4))
    plt.scatter(x_zoom, y_zoom, alpha=0.7)
    plt.xlabel("Depth (microns)")
    plt.ylabel("Fibrosis (%)")
    plt.title(title)
    plt.grid(True)
    plt.show()

# ---------------------------
# Choose regions to zoom
# ---------------------------

# Shallow region
plot_zoom(x, y, 0, 2000, "Zoom: Shallow Depth (0–2000 microns)")
# Add trend line to zoom

# Middle region
plot_zoom(x, y, 2000, 6000, "Zoom: Mid Depth (2000–6000 microns)")

# Deep region
plot_zoom(x, y, 6000, 10000, "Zoom: Deep Depth (6000–10000 microns)")



#%%
# ---------------------------
# Interpolation comparison by region
# ---------------------------
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compare_interpolation_region_all_points(df, xmin, xmax, region_name):
    # Select region
    region_df = df[
        (df["Depth (microns)"] >= xmin) &
        (df["Depth (microns)"] <= xmax)
    ].copy()

    # Combine duplicate depths by averaging White percent
    region_df = (
        region_df.groupby("Depth (microns)", as_index=False)["White percent"]
        .mean()
        .sort_values(by="Depth (microns)")
        .reset_index(drop=True)
    )

    x_all = region_df["Depth (microns)"].values
    y_all = region_df["White percent"].values

    print(f"\n--- {region_name} Region ---")
    print(f"Total unique depth points in region: {len(region_df)}")

    if len(region_df) < 6:
        print("Not enough unique points in this region for interpolation comparison.")
        return None, None

    # Keep every other point
    train_df = region_df.iloc[::2].copy()
    test_df = region_df.iloc[1::2].copy()

    x_train = train_df["Depth (microns)"].values
    y_train = train_df["White percent"].values
    x_test = test_df["Depth (microns)"].values
    y_test = test_df["White percent"].values

    # Only test points inside interpolation range
    valid_mask = (x_test >= x_train.min()) & (x_test <= x_train.max())
    x_test = x_test[valid_mask]
    y_test = y_test[valid_mask]

    if len(x_test) == 0:
        print("No removed points fell inside the interpolation range.")
        return None, None

    # Linear interpolation
    linear_interp = interp1d(x_train, y_train, kind="linear")
    y_linear_pred = linear_interp(x_test)

    # Quadratic interpolation
    quadratic_possible = len(np.unique(x_train)) >= 3
    if quadratic_possible:
        quadratic_interp = interp1d(x_train, y_train, kind="quadratic")
        y_quad_pred = quadratic_interp(x_test)
    else:
        quadratic_interp = None
        y_quad_pred = np.full_like(y_test, np.nan, dtype=float)

    # Build comparison table for ALL removed points
    compare_df = pd.DataFrame({
        "Depth (microns)": x_test,
        "Actual fibrosis (%)": y_test,
        "Linear interpolation (%)": y_linear_pred,
        "Quadratic interpolation (%)": y_quad_pred,
        "Linear abs error": np.abs(y_linear_pred - y_test),
        "Quadratic abs error": np.abs(y_quad_pred - y_test)
    })

    # Percent errors
    compare_df["Linear percent error"] = np.where(
        compare_df["Actual fibrosis (%)"] != 0,
        compare_df["Linear abs error"] / compare_df["Actual fibrosis (%)"] * 100,
        np.nan
    )
    compare_df["Quadratic percent error"] = np.where(
        compare_df["Actual fibrosis (%)"] != 0,
        compare_df["Quadratic abs error"] / compare_df["Actual fibrosis (%)"] * 100,
        np.nan
    )

    print("\nComparison table for ALL removed points:")
    print(compare_df)

    # Reconstruct full datasets:
    # 1) original full data
    # 2) linear reconstruction = kept points + linear predictions
    # 3) quadratic reconstruction = kept points + quadratic predictions

    linear_reconstructed_df = pd.DataFrame({
        "Depth (microns)": np.concatenate([x_train, x_test]),
        "Fibrosis (%)": np.concatenate([y_train, y_linear_pred])
    }).sort_values(by="Depth (microns)")

    quadratic_reconstructed_df = pd.DataFrame({
        "Depth (microns)": np.concatenate([x_train, x_test]),
        "Fibrosis (%)": np.concatenate([y_train, y_quad_pred])
    }).sort_values(by="Depth (microns)")

    original_df = pd.DataFrame({
        "Depth (microns)": x_all,
        "Fibrosis (%)": y_all
    }).sort_values(by="Depth (microns)")

    # Plot all three curves together
    plt.figure(figsize=(9, 6))

    # Original full dataset
    plt.plot(
        original_df["Depth (microns)"],
        original_df["Fibrosis (%)"],
        marker="o",
        linestyle="-",
        linewidth=2,
        label="Original data"
    )

    # Linear reconstructed dataset
    plt.plot(
        linear_reconstructed_df["Depth (microns)"],
        linear_reconstructed_df["Fibrosis (%)"],
        marker="x",
        linestyle="-",
        linewidth=2,
        label="Linear interpolation + kept points"
    )

    # Quadratic reconstructed dataset
    if quadratic_possible:
        plt.plot(
            quadratic_reconstructed_df["Depth (microns)"],
            quadratic_reconstructed_df["Fibrosis (%)"],
            marker="^",
            linestyle="-",
            linewidth=2,
            label="Quadratic interpolation + kept points"
        )

    plt.xlabel("Depth (microns)")
    plt.ylabel("Fibrosis (%)")
    plt.title(f"{region_name} Region: Original vs Reconstructed Data")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Summary metrics
    summary = {
        "Region": region_name,
        "Linear mean abs error": compare_df["Linear abs error"].mean(),
        "Quadratic mean abs error": compare_df["Quadratic abs error"].mean(),
        "Linear mean percent error": compare_df["Linear percent error"].mean(),
        "Quadratic mean percent error": compare_df["Quadratic percent error"].mean()
    }

    return compare_df, summary


# ---------------------------
# Run interpolation comparison for each region
# ---------------------------
shallow_compare, shallow_summary = compare_interpolation_region_all_points(
    results_df, 0, 2000, "Shallow"
)

mid_compare, mid_summary = compare_interpolation_region_all_points(
    results_df, 2000, 6000, "Mid"
)

deep_compare, deep_summary = compare_interpolation_region_all_points(
    results_df, 6000, 10000, "Deep"
)

# ---------------------------
# Summary table
# ---------------------------
summary_df = pd.DataFrame([
    shallow_summary,
    mid_summary,
    deep_summary
])

print("\nAverage interpolation error summary:")
print(summary_df)


#%%

# ---------------------------
# Three-point interpolation by region
# Keep first, median, and last points
# ---------------------------
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def three_point_interpolation_region(df, xmin, xmax, region_name):
    # Select region
    region_df = df[
        (df["Depth (microns)"] >= xmin) &
        (df["Depth (microns)"] <= xmax)
    ].copy()

    # Combine duplicate depths by averaging White percent
    region_df = (
        region_df.groupby("Depth (microns)", as_index=False)["White percent"]
        .mean()
        .sort_values(by="Depth (microns)")
        .reset_index(drop=True)
    )

    x_all = region_df["Depth (microns)"].values
    y_all = region_df["White percent"].values

    print(f"\n--- {region_name} Region ---")
    print(f"Total unique depth points in region: {len(region_df)}")

    if len(region_df) < 5:
        print("Not enough points in this region for three-point interpolation.")
        return None, None

    # Keep first, median, and last points
    mid_index = len(region_df) // 2
    keep_indices = sorted(list(set([0, mid_index, len(region_df) - 1])))

    train_df = region_df.iloc[keep_indices].copy()
    test_df = region_df.drop(index=keep_indices).copy()

    x_train = train_df["Depth (microns)"].values
    y_train = train_df["White percent"].values
    x_test = test_df["Depth (microns)"].values
    y_test = test_df["White percent"].values

    # Only keep test points inside interpolation range
    valid_mask = (x_test >= x_train.min()) & (x_test <= x_train.max())
    x_test = x_test[valid_mask]
    y_test = y_test[valid_mask]

    if len(x_test) == 0:
        print("No removed points fell inside the interpolation range.")
        return None, None

    # Linear interpolation
    linear_interp = interp1d(x_train, y_train, kind="linear")
    y_linear_pred = linear_interp(x_test)

    # Quadratic interpolation
    quadratic_interp = interp1d(x_train, y_train, kind="quadratic")
    y_quad_pred = quadratic_interp(x_test)

    # Build comparison table for all removed points
    compare_df = pd.DataFrame({
        "Depth (microns)": x_test,
        "Actual fibrosis (%)": y_test,
        "Linear interpolation (%)": y_linear_pred,
        "Quadratic interpolation (%)": y_quad_pred,
        "Linear abs error": np.abs(y_linear_pred - y_test),
        "Quadratic abs error": np.abs(y_quad_pred - y_test)
    })

    compare_df["Linear percent error"] = np.where(
        compare_df["Actual fibrosis (%)"] != 0,
        compare_df["Linear abs error"] / compare_df["Actual fibrosis (%)"] * 100,
        np.nan
    )

    compare_df["Quadratic percent error"] = np.where(
        compare_df["Actual fibrosis (%)"] != 0,
        compare_df["Quadratic abs error"] / compare_df["Actual fibrosis (%)"] * 100,
        np.nan
    )

    print("\nComparison table for all removed points:")
    print(compare_df)

    # Reconstructed datasets
    original_df = pd.DataFrame({
        "Depth (microns)": x_all,
        "Fibrosis (%)": y_all
    }).sort_values(by="Depth (microns)")

    linear_reconstructed_df = pd.DataFrame({
        "Depth (microns)": np.concatenate([x_train, x_test]),
        "Fibrosis (%)": np.concatenate([y_train, y_linear_pred])
    }).sort_values(by="Depth (microns)")

    quadratic_reconstructed_df = pd.DataFrame({
        "Depth (microns)": np.concatenate([x_train, x_test]),
        "Fibrosis (%)": np.concatenate([y_train, y_quad_pred])
    }).sort_values(by="Depth (microns)")

    # Plot
    plt.figure(figsize=(9, 6))

    plt.plot(
        original_df["Depth (microns)"],
        original_df["Fibrosis (%)"],
        marker="o",
        linestyle="-",
        linewidth=2,
        label="Original data"
    )

    plt.plot(
        linear_reconstructed_df["Depth (microns)"],
        linear_reconstructed_df["Fibrosis (%)"],
        marker="x",
        linestyle="-",
        linewidth=2,
        label="Linear interpolation + kept points"
    )

    plt.plot(
        quadratic_reconstructed_df["Depth (microns)"],
        quadratic_reconstructed_df["Fibrosis (%)"],
        marker="^",
        linestyle="-",
        linewidth=2,
        label="Quadratic interpolation + kept points"
    )

    plt.scatter(
        x_train,
        y_train,
        s=110,
        marker="s",
        label="Kept points (first, median, last)"
    )

    plt.xlabel("Depth (microns)")
    plt.ylabel("Fibrosis (%)")
    plt.title(f"{region_name} Region: Three-Point Interpolation Comparison")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Summary
    summary = {
        "Region": region_name,
        "Linear mean abs error": compare_df["Linear abs error"].mean(),
        "Quadratic mean abs error": compare_df["Quadratic abs error"].mean(),
        "Linear mean percent error": compare_df["Linear percent error"].mean(),
        "Quadratic mean percent error": compare_df["Quadratic percent error"].mean()
    }

    return compare_df, summary


# ---------------------------
# Run three-point comparison for each region
# ---------------------------
shallow_three_compare, shallow_three_summary = three_point_interpolation_region(
    results_df, 0, 2000, "Shallow"
)

mid_three_compare, mid_three_summary = three_point_interpolation_region(
    results_df, 2000, 6000, "Mid"
)

deep_three_compare, deep_three_summary = three_point_interpolation_region(
    results_df, 6000, 10000, "Deep"
)

# ---------------------------
# Summary table
# ---------------------------
three_point_summary_df = pd.DataFrame([
    shallow_three_summary,
    mid_three_summary,
    deep_three_summary
])

print("\nThree-point interpolation error summary:")
print(three_point_summary_df)

# %%
