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
plot_zoom(x, y, 6000, 10 000, "Zoom: Deep Depth (6000–10000 microns)")







# %%

# ---------------------------
# Interpolation comparison by region
# ---------------------------
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compare_interpolation_region(df, xmin, xmax, region_name, num_compare_points=5):
    region_df = df[
        (df["Depth (microns)"] >= xmin) &
        (df["Depth (microns)"] <= xmax)
    ].copy()

    # Combine duplicate depths by averaging white percent
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

    if len(region_df) < 8:
        print("Not enough unique points in this region for comparison.")
        return None

    # Keep every other point for interpolation
    train_df = region_df.iloc[::2].copy()
    test_df = region_df.iloc[1::2].copy()

    x_train = train_df["Depth (microns)"].values
    y_train = train_df["White percent"].values
    x_test = test_df["Depth (microns)"].values
    y_test = test_df["White percent"].values

    # Only keep test points inside interpolation range
    valid_mask = (x_test >= x_train.min()) & (x_test <= x_train.max())
    x_test = x_test[valid_mask]
    y_test = y_test[valid_mask]

    if len(x_test) < num_compare_points:
        print(f"Only {len(x_test)} valid removed points available.")
        num_compare_points = len(x_test)

    if len(x_test) == 0:
        print("No valid removed points in interpolation range.")
        return None

    # Linear interpolation always attempted
    linear_interp = interp1d(x_train, y_train, kind="linear")
    y_linear_pred = linear_interp(x_test)

    # Quadratic interpolation only if enough unique points
    quadratic_possible = len(np.unique(x_train)) >= 3

    if quadratic_possible:
        quadratic_interp = interp1d(x_train, y_train, kind="quadratic")
        y_quad_pred = quadratic_interp(x_test)
    else:
        quadratic_interp = None
        y_quad_pred = np.full_like(y_test, np.nan, dtype=float)

    # Pick 5 evenly spaced removed points
    compare_indices = np.linspace(0, len(x_test) - 1, num_compare_points, dtype=int)

    x_compare = x_test[compare_indices]
    y_actual_compare = y_test[compare_indices]
    y_linear_compare = y_linear_pred[compare_indices]
    y_quad_compare = y_quad_pred[compare_indices]

    compare_df = pd.DataFrame({
        "Depth (microns)": x_compare,
        "Actual fibrosis (%)": y_actual_compare,
        "Linear interpolation (%)": y_linear_compare,
        "Quadratic interpolation (%)": y_quad_compare,
        "Linear abs error": np.abs(y_linear_compare - y_actual_compare),
        "Quadratic abs error": np.abs(y_quad_compare - y_actual_compare)
    })

    print("\nComparison table:")
    print(compare_df)

    # Smooth interpolation curves
    x_line = np.linspace(x_train.min(), x_train.max(), 400)
    y_line_linear = linear_interp(x_line)

    if quadratic_interp is not None:
        y_line_quad = quadratic_interp(x_line)

    # Sort comparison points so lines connect correctly
    sort_idx = np.argsort(x_compare)
    x_compare_sorted = x_compare[sort_idx]
    y_actual_sorted = y_actual_compare[sort_idx]
    y_linear_sorted = y_linear_compare[sort_idx]
    y_quad_sorted = y_quad_compare[sort_idx]

    plt.figure(figsize=(8, 5))
    plt.scatter(x_all, y_all, alpha=0.6, label="Actual plotted points")
    plt.plot(x_line, y_line_linear, linewidth=2, label="Linear interpolation")

    if quadratic_interp is not None:
        plt.plot(x_line, y_line_quad, linewidth=2, label="Quadratic interpolation")

    plt.scatter(x_compare_sorted, y_actual_sorted, s=80, marker="o", label="5 actual comparison points")
    plt.plot(
        x_compare_sorted,
        y_linear_sorted,
        marker="x",
        linestyle='-',
        linewidth=2,
        label="5 linear estimates"
    )

    if quadratic_interp is not None:
        plt.plot(
            x_compare_sorted,
            y_quad_sorted,
            marker="^",
            linestyle='-',
            linewidth=2,
            label="5 quadratic estimates"
        )

    plt.xlabel("Depth (microns)")
    plt.ylabel("Fibrosis (%)")
    plt.title(f"{region_name} Region: Actual vs Interpolated Points")
    plt.grid(True)
    plt.legend()
    plt.show()

    return compare_df

# ---------------------------
# Run interpolation comparison for each region
# ---------------------------
shallow_compare = compare_interpolation_region(results_df, 0, 2000, "Shallow", num_compare_points=5)
mid_compare = compare_interpolation_region(results_df, 2000, 6000, "Mid", num_compare_points=5)
deep_compare = compare_interpolation_region(results_df, 6000, 10000, "Deep", num_compare_points=5)

# ---------------------------
# Summary of interpolation errors
# ---------------------------
def summarize_errors(compare_df, region_name):
    if compare_df is None:
        return {
            "Region": region_name,
            "Mean linear abs error": np.nan,
            "Mean quadratic abs error": np.nan
        }

    return {
        "Region": region_name,
        "Mean linear abs error": compare_df["Linear abs error"].mean(),
        "Mean quadratic abs error": compare_df["Quadratic abs error"].mean()
    }

summary_list = [
    summarize_errors(shallow_compare, "Shallow"),
    summarize_errors(mid_compare, "Mid"),
    summarize_errors(deep_compare, "Deep")
]

summary_df = pd.DataFrame(summary_list)
print("\nAverage interpolation error summary:")
print(summary_df)
# %%
