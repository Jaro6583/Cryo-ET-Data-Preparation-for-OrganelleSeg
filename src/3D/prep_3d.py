import argparse
import os
import numpy as np
import mrcfile
import tifffile
from scipy.ndimage import zoom


def get_mrc_metadata(path):
    """Automatically extracts voxel size from the MRC/REC header."""
    with mrcfile.open(path, permissive=True) as mrc:
        # Returns (z_spacing, y_spacing, x_spacing)
        return mrc.voxel_size.x


def pad_to_divisible(data, divisor=32):
    # Pads the 3D volume with zeros to make dimensions divisible by 32.
    z, y, x = data.shape
    new_z = int(np.ceil(z / divisor) * divisor)
    new_y = int(np.ceil(y / divisor) * divisor)
    new_x = int(np.ceil(x / divisor) * divisor)

    pad_z = new_z - z
    pad_y = new_y - y
    pad_x = new_x - x

    # Distribute padding (adds mostly to the end of dimensions)
    padded_data = np.pad(data, ((0, pad_z), (0, pad_y), (0, pad_x)),
                         mode='constant')
    return padded_data


def process_3d_pair(img_path, lbl_path, target_spacing, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(img_path).split('.')[0]

    # 1. Load data
    with mrcfile.open(img_path, permissive=True) as m_img:
        with mrcfile.open(lbl_path, permissive=True) as m_lbl:
            img_data = m_img.data.astype(np.float32)
            lbl_data = m_lbl.data.astype(np.uint8)
            current_spacing = m_img.voxel_size.x
    
    print(f"--- Processing {base_name} ---")
    print(f"Current spacing: {current_spacing:.2f} A/px |" +
          f"Target: {target_spacing:.2f} A/px")

    # 2. Rescale (Zoom)
    if not np.isclose(current_spacing, target_spacing):
        scale_factor = current_spacing / target_spacing
        print(f"Rescaling by factor: {scale_factor:.3f}")

        # Order 3 = Spline interpolation for images (smooth)
        img_data = zoom(img_data, scale_factor, order=3)
        # Order 0 = Nearest neighbor for labels (preserves integer classes)
        lbl_data = zoom(lbl_data, scale_factor, order=0)
    
    # 3. Pad to divisible by 32
    print(f"Shape after scaling: {img_data.shape}")
    img_data = pad_to_divisible(img_data)
    lbl_data = pad_to_divisible(lbl_data)
    print(f"Final padded shape: {img_data.shape} (Divisible by 32)")

    # 4. Save as 3D TIFF
    img_out = os.path.join(output_dir, f"{base_name}_struct.tif")
    lbl_out = os.path.join(output_dir, f"{base_name}_label.tif")

    # imagej=True allows standard viewers to see it as a stack
    tifffile.imwrite(img_out, img_data, imagej=True)
    tifffile.imwrite(lbl_out, lbl_data, imagej=True)
    print(f"Saved: {img_out}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prep 3D Cryo-ET pairs for training."
    )
    parser.add_argument("--image", required=True,
                        help="Path to 3D .mrc or .rec structure")
    parser.add_argument("--label", required=True,
                        help="Path to 3D .mrc label/mask")
    parser.add_argument("--target_spacing", type=float, required=True,
                        help="Desired Angstroms per pixel.")
    parser.add_argument("--output", default="prepped_3d_data",
                        help="Output directory")

    args = parser.parse_args()
    process_3d_pair(args.image, args.label, args.target_spacing, args.output)
