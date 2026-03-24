'''Module 3: count black and white pixels and compute the percentage of white pixels in a .jpg image and extrapolate points'''
#%%
# %%
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy.interpolate import interp1d

# --------------------------------------------------
# 5 image files and their depths from the CSV
# --------------------------------------------------

image_info = [
    ("C:\\Users\\15712\\OneDrive - University of Virginia\\Comp Mod 3\\Module-3-Fibrosisb-breckley_elfaki\\images\\MASK_Sk658 Llobe ch010021.jpg", 30),
    ("C:\\Users\\15712\\OneDrive - University of Virginia\\Comp Mod 3\\Module-3-Fibrosisb-breckley_elfaki\\images\\MASK_Sk658 Llobe ch010017.jpg", 45),
    ("C:\\Users\\15712\\OneDrive - University of Virginia\\Comp Mod 3\\Module-3-Fibrosisb-breckley_elfaki\\images\\MASK_Sk658 Llobe ch010019.jpg", 60),
    ("C:\\Users\\15712\\OneDrive - University of Virginia\\Comp Mod 3\\Module-3-Fibrosisb-breckley_elfaki\\images\\MASK_Sk658 Llobe ch010022.jpg", 80),
    ("C:\\Users\\15712\\OneDrive - University of Virginia\\Comp Mod 3\\Module-3-Fibrosisb-breckley_elfaki\\images\\MASK_Sk658 Llobe ch010018.jpg", 90),
    ("C:\\Users\\15712\\OneDrive - University of Virginia\\Comp Mod 3\\Module-3-Fibrosisb-breckley_elfaki\\images\\MASK_Sk658 Llobe ch010023.jpg", 100),

]

threshold_value = 127
results = []

print("Counts of pixels by color in each image\n")

# --------------------------------------------------
# Process each image one at a time
# --------------------------------------------------

for filepath, depth in image_info:
    path = Path(filepath)

    if not path.exists():
        print(f"File not found: {path}")
        continue

    # Open image and convert to grayscale
    img = Image.open(path).convert("L")
    img_array = np.array(img)

    # Make binary image:
    # pixels > threshold are white, the rest are black
    binary = img_array > threshold_value

    white_count = np.count_nonzero(binary)
    total_pixels = binary.size
    black_count = total_pixels - white_count
    white_percent = 100 * white_count / total_pixels

    results.append({
        "Filename": path.name,
        "Depth (microns)": depth,
        "White pixels": white_count,
        "Black pixels": black_count,
        "White percent": white_percent
    })

    print(f"{path.name}")
    print(f"White pixels: {white_count}")
    print(f"Black pixels: {black_count}")
    print(f"{white_percent:.2f}% White | Depth: {depth} microns\n")

# --------------------------------------------------
# Save results to CSV
# --------------------------------------------------

df = pd.DataFrame(results)
df.to_csv("Percent_White_Pixels.csv", index=False)

print("The .csv file 'Percent_White_Pixels.csv' has been created.")

# --------------------------------------------------
# Plot depth vs percent white
# --------------------------------------------------

depths = df["Depth (microns)"].to_numpy()
white_percents = df["White percent"].to_numpy()

plt.figure(figsize=(8, 5))
plt.scatter(depths, white_percents)
plt.plot(depths, white_percents)
plt.title("Depth of Image vs Percentage of White Pixels")
plt.xlabel("Depth of image (microns)")
plt.ylabel("White pixels as % of total pixels")
plt.grid(True)
plt.tight_layout()
plt.show()

# --------------------------------------------------
# Optional interpolation
# Uncomment if needed
# --------------------------------------------------

# interpolate_depth = float(input("Enter the depth to interpolate (in microns): "))
# interpolator = interp1d(depths, white_percents, kind="linear")
# interpolated_percent = float(interpolator(interpolate_depth))

# print(
#     f"The interpolated point is at depth {interpolate_depth} microns "
#     f"with white pixel percentage {interpolated_percent:.2f}%."
# )

# plt.figure(figsize=(8, 5))
# plt.scatter(depths, white_percents, label="Original data")
# plt.plot(depths, white_percents)
# plt.scatter(interpolate_depth, interpolated_percent, color="red", s=80, label="Interpolated point")
# plt.title("Depth vs Percentage of White Pixels with Interpolated Point")
# plt.xlabel("Depth of image (microns)")
# plt.ylabel("White pixels as % of total pixels")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()


##############
# LECTURE 2: UNCOMMENT BELOW

# # Interpolate a point: given a depth, find the corresponding white pixel percentage

# interpolate_depth = float(input(colored(
#     "Enter the depth at which you want to interpolate a point (in microns): ", "yellow")))

# x = depths
# y = white_percents

# # You can also use 'quadratic', 'cubic', etc.
# i = interp1d(x, y, kind='linear')
# interpolate_point = i(interpolate_depth)
# print(colored(
#     f'The interpolated point is at the x-coordinate {interpolate_depth} and y-coordinate {interpolate_point}.', "green"))

# depths_i = depths[:]
# depths_i.append(interpolate_depth)
# white_percents_i = white_percents[:]
# white_percents_i.append(interpolate_point)


# # make two plots: one that doesn't contain the interpolated point, just the data calculated from your images, and one that also contains the interpolated point (shown in red)
# fig, axs = plt.subplots(2, 1)

# axs[0].scatter(depths, white_percents, marker='o', linestyle='-', color='blue')
# axs[0].set_title('Plot of depth of image vs percentage white pixels')
# axs[0].set_xlabel('depth of image (in microns)')
# axs[0].set_ylabel('white pixels as a percentage of total pixels')
# axs[0].grid(True)


# axs[1].scatter(depths_i, white_percents_i, marker='o',
#                linestyle='-', color='blue')
# axs[1].set_title(
#     'Plot of depth of image vs percentage white pixels with interpolated point (in red)')
# axs[1].set_xlabel('depth of image (in microns)')
# axs[1].set_ylabel('white pixels as a percentage of total pixels')
# axs[1].grid(True)
# axs[1].scatter(depths_i[len(depths_i)-1], white_percents_i[len(white_percents_i)-1],
#                color='red', s=100, label='Highlighted point')


# # Adjust layout to prevent overlap
# plt.tight_layout()
# plt.show()

# %%
