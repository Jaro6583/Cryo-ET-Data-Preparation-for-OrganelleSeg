import argparse
import zarr
import numpy as np
import pandas as pd
import sys
import os


def parse_args():
    msg = "Generate binary Zarr masks for Ribosomes "
    msg += "and FAS in CellMap format."
    parser = argparse.ArgumentParser(
        description=msg
    )

    # Paths matching the CellMap layout
    msg = "Path to the EM array "
    msg += "(e.g., .../recon-1/em/fibsem-uint8/s0) to match shape/scale."
    parser.add_argument(
        "--em-zarr",
        required=True,
        help=msg
    )
    msg = 'Path to the groundtruth crop directory '
    msg += '(e.g., .../labels/groundtruth/crop1).'
    parser.add_argument(
        "--gt-dir",
        required=True,
        help=msg
    )

    parser.add_argument(
        "--ribosome-csv",
        required=True,
        help="Path to headerless XYZ Ribosome coordinates CSV."
    )
    parser.add_argument(
        "--fas-csv",
        required=True,
        help="Path to headerless XYZ FAS coordinates CSV."
    )

    # Naming the classes for the output folders
    parser.add_argument("--ribo-class-name", type=str, default='ribosomes',
                        help='Folder name for ribosomes (default: ribosomes)')
    parser.add_argument(
        "--fas-class-name",
        type=str,
        default='fas',
        help="Folder name for FAS (default: fas)"
    )

    # Biological sizes in Angstroms
    parser.add_argument(
        "--ribo-diameter-A",
        type=float,
        default=300.0,
        help="Ribosome diameter in Angstroms (default: 300)."
    )
    parser.add_argument(
        "--fas-diameter-A",
        type=float,
        default=250.0,
        help="FAS diameter in Angstroms (default: 250)."
    )

    # Metadata overrides
    msg = 'Pixel spacing in Angstroms/pixel. '
    msg += 'If not provided, script attempts to read from EM Zarr metadata.'
    parser.add_argument(
        "--pixel-spacing",
        type=float,
        default=None,
        help=msg
    )

    return parser.parse_args()


def extract_pixel_spacing(em_zarr_path):
    # Attempt to extract pixel spacing from the parent OME-Zarr group.
    # (The scale is usually stored in the parent group: .zatters of the 'fibsem-uint8' folder)  # noqa
    parent_group_path = os.path.dirname(em_zarr_path.rstrip('/'))
    try:
        grp = zarr.open(parent_group_path, mode="r")
        attrs = grp.attrs.asdict()
        if "multiscales" in attrs:
            datasets = attrs["multiscales"][0]["datasets"]
            for transform in datasets[0].get("coordinateTransformations", []):
                if transform["type"] == "scale":
                    # Return the x-scale (assuming isotropic)
                    return float(transform["scale"][-1])
    except Exception:
        pass
    return None


def draw_sphere(volume, center_z, center_y, center_x, radius):
    # Draws a solid 3D sphere of 1s into the binary volume.
    z, y, x = ind(center_z), ind(center_y), ind(center_x)
    r = int(np.ceil(radius))

    z_min, z_max = max(0, z-r), min(volume.shape[0], z + r + 1)
    y_min, y_max = max(0, y-r), min(volume.shape[1], y + r + 1)
    x_min, x_max = max(0, x-r), min(volume.shape[2], x + r + 1)

    zz, yy, xx = np.ogrid[z_min:z_max, y_min:y_max, x_min:x_max]
    dist_sq = (zz - z)**2 + (yy - y)**2 + (xx - x)**2

    mask = dist_sq <= radius**2
    volume[z_min:z_max, y_min:y_max, x_min:x_max][mask] = 1


def generate_class_mask(shape, csv_path, radius_pixels, out_dir, class_name):
    # Generates a binary mask array and saves it as s0 in the class directory
    if not os.path.exists(csv_path):
        print(f"Warning: CSV not found at {csv_path}. Skipping {class_name}.")
        return

    print(f"\nProcessing {class_name}...")
    volume = np.zeros(shape, dtype=np.uint8)

    df = pd.read_csv(csv_path, header=None,
                     usecols=[0, 1, 2], names=['x', 'y', 'z'])
    original_len = len(df)

    df['x'] = pd.to_numeric(df['x'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df['z'] = pd.to_numeric(df['z'], errors='coerce')
    df = df.dropna(subset=['x', 'y', 'z'])

    skipped = original_len - len(df)
    if skipped > 0:
        print(f"  -> Skipped {skipped} invalid rows.")

    print(f"  -> Drawing {len(df)} spheres of radius {radius_pixels:.2f}",
          "px...")
    for _, row in df.iterrows():
        draw_sphere(volume, row['z'], row['y'], row['x'], radius_pixels)

    # CellMap format: groundtruth/{crop}/{class_name}/s0
    out_path = os.path.join(out_dir, class_name, "s0")
    print(f"  -> Saving to {out_path}...")
    zarr.save(out_path, volume, chunks=(64, 64, 64))


def main():
    args = parse_args()

    # 1. Read EM array shape
    print(f"Opening EM Zarr to determine shape: {args.em_zarr}")
    em_array = zarr.open(args.em_zarr, mode='r')
    shape = em_array.shape
    print(f"Tomogram shape detected: {shape}")

    # 2. Determine pixel spacing
    pixel_spacing = args.pixel_spacing
    if pixel_spacing is None:
        pixel_spacing = extract_pixel_spacing(args.em_zarr)
        if pixel_spacing is None:
            print("Error: Could not determine pixel spacing from metadata.",
                  "Use --pixel-spacing.")
            sys.exit(1)
        print(f"Auto-detected pixel spacing: {pixel_spacing} Angstroms/pixel")
    else:
        print(f"Using manual pixel spacing: {pixel_spacing} Angstroms/pixel")

    # 3. Calculate Radii
    ribo_radius_px = (args.ribo_diameter_A / 2.0) / pixel_spacing
    fas_radius_px = (args.fas_diameter_A / 2.0) / pixel_spacing

    # 4. Ensure groundtruth output directory exists
    os.makedirs(args.gt_dir, exist_ok=True)

    # 5. Generate and save independent binary masks
    generate_class_mask(shape, args.ribosome_csv, ribo_radius_px,
                        args.gt_dir, args.ribo_class_name)
    generate_class_mask(shape, args.fas_csv, fas_radius_px,
                        args.gt_dir, args.fas_class_name)

    print("\nAll done! The new classes are ready for the zarr_loader.")


if __name__ == "__main__":
    main()
