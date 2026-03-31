import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import mrcfile
import tifffile
from tqdm import tqdm


def normalize_image(image):
    """Linearly scales a 32-bit float image to 0-255 (8-bit) for the NN."""
    p2, p98 = np.percentile(image, (2, 98))
    image = np.clip(image, p2, p98)
    image = (image - p2) / (p98 - p2)
    return (image * 255).astype(np.uint8)


def get_tile_indicies(image_size, tile_size, num_tiles):
    """Calculates the starting indicies for overlapping tiles."""
    if num_tiles == 1:
        return [0]
    stride = (image_size - tile_size) // (num_tiles - 1)
    return [i * stride for i in range(num_tiles)]


def colorize_labels(label_array, num_classes=15):
    # Maps 0-14 integers to distinct RGB colors for human visualization.
    cmap = plt.get_cmap('tab20', num_classes)
    # Map the integers to colors
    colored = cmap(label_array)
    return (colored[:, :, :3] * 255).astype(np.uint8)


def process_tomogram(image_path,
                     label_path,
                     output_dir,
                     tile_size,
                     grid_size,
                     slice_step,
                     filter_empty):
    # Ensure output directories exist
    img_out = os.path.join(output_dir, "images")
    lbl_out = os.path.join(output_dir, "labels")
    pre_out = os.path.join(output_dir, "previews")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)
    os.makedirs(pre_out, exist_ok=True)

    base_name = os.path.basename(image_path).split('.')[0]

    with mrcfile.open(image_path, permissive=True) as mrc_img:
        with mrcfile.open(label_path, permissive=True) as mrc_lbl:

            data_img = mrc_img.data
            data_lbl = mrc_lbl.data

            num_slices = data_img.shape[0]
            dim_y, dim_x = data_img.shape[1], data_img.shape[2]

            # Calculate tile start positions
            y_starts = get_tile_indicies(dim_y, tile_size, grid_size)
            x_starts = get_tile_indicies(dim_x, tile_size, grid_size)

            print(f"Processing {base_name}: {num_slices} slices,"
                  + f"extracting every {slice_step}...")

            for z in tqdm(range(0, num_slices, slice_step)):
                slice_img = data_img[z]
                slice_lbl = data_lbl[z]

                # Pre-normalize the whole slice for consistency
                slice_img_norm = normalize_image(slice_img)

                for row, y in enumerate(y_starts):
                    for col, x in enumerate(x_starts):
                        # Define crop boundaries
                        img_crop = slice_img_norm[y:y+tile_size,
                                                  x:x+tile_size]
                        lbl_crop = slice_lbl[y:y+tile_size, x:x+tile_size]

                        # If the flag is set and the label array is entirely 0
                        if filter_empty and not np.any(lbl_crop):
                            continue

                        # Naming convention: prefix_z000_r0_c0.tif
                        tile_id = f"{base_name}_z{z:03d}_r{row}_c{col}.tif"

                        # Save TIFFs
                        tifffile.imwrite(os.path.join(img_out, tile_id),
                                         img_crop)
                        tifffile.imwrite(os.path.join(lbl_out, tile_id),
                                         lbl_crop.astype(np.uint8))
                        lbl_preview = colorize_labels(lbl_crop)
                        tifffile.imwrite(os.path.join(pre_out, tile_id),
                                         lbl_preview)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Crop 3D MRC volumes into 2D overlapping TIFF crops."
    )
    parser.add_argument("--image", required=True,
                        help="Path to the 3D .mrc or .rec image file")
    parser.add_argument("--label", required=True,
                        help="Path to the 3D .mrc annotation file")
    parser.add_argument("--output", default="training_data",
                        help="Output directory")
    parser.add_argument("--crop_size", type=int, default=256,
                        help="Square crop size in pixels")
    parser.add_argument(
        "--grid_size",
        type=int,
        default=4,
        help="Number of crops per dimension (e.g. 4 for 4x4=16)"
    )
    parser.add_argument("--step", type=int, default=5,
                        help="Analyze every Nth slice")
    helpmsg = "Skip saving tiles that contain only background (all 0s)."
    parser.add_argument("--filter_empty", default=False, help=helpmsg)

    args = parser.parse_args()
    process_tomogram(args.image,
                     args.label,
                     args.output,
                     args.crop_size,
                     args.grid_size,
                     args.step,
                     args.filter_empty)
