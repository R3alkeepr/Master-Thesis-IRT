import os
import glob

import imageio.v2 as imageio  # pip install imageio
from PIL import Image         # pip install pillow


# Folder with your plots (kept identical)
plot_folder = r"F:/Masterarbeit_Backup_2/Plots_aligned_angled/Paradiestal_12-13.08.25"

# New folder for GIFs (kept identical)
gif_folder = r"F:/Masterarbeit_Backup_2/GIFs_aligned_angled"
os.makedirs(gif_folder, exist_ok=True)

# Get all PNG files (sorted by filename; kept identical)
png_files = sorted(glob.glob(os.path.join(plot_folder, "*.png")))
if not png_files:
    raise FileNotFoundError(f"No PNG files found in: {plot_folder}")

# Read first image to get dimensions (kept identical intent)
with Image.open(png_files[0]) as first_img:
    img_width, img_height = first_img.size  # PIL gives (width, height)

# Output GIF path (kept identical base name; R produced "Karwendel_11-12.11.25" without extension)
gif_file = os.path.join(gif_folder, "Paradiestal_12-13.08.25.gif")

# Total duration = 10 seconds (R comment says 12, but code uses 10; we match code)
n_frames = len(png_files)
duration_per_frame = 10.0 / n_frames  # seconds per frame

# Read frames, enforce identical size (same as R passing width/height)
frames = []
for fp in png_files:
    with Image.open(fp) as im:
        # Ensure every frame is exactly the same size as the first frame
        if im.size != (img_width, img_height):
            im = im.resize((img_width, img_height), resample=Image.Resampling.LANCZOS)
        frames.append(im.convert("RGBA"))

# Write GIF (loop forever)
imageio.mimsave(
    gif_file,
    frames,
    duration=duration_per_frame,  # seconds per frame
    loop=0,                       # 0 = infinite loop
)

print(f"Saved GIF: {gif_file}")
print(f"Frames: {n_frames} | Duration/frame: {duration_per_frame:.4f}s | Total: {duration_per_frame * n_frames:.2f}s")
