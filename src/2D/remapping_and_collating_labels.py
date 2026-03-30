import os
import argparse
import json
import numpy as np
import mrcfile


def load_class_mapping(json_path):
    # Loads the JSON and creates a flexible lookup dictionary.
    with open(json_path, "r") as f:
        data = json.load(f)

    classes = data.get('classes', {})
    lookup = {}

    # Allow the user to type either the string name or the integer ID
    for name, class_id in classes.items():
        lookup[name.lower()] = class_id
        lookup[str(class_id)] = class_id

    return lookup, classes


def collate_annotations(input_files, json_path, output_filepath):
    lookup, class_dict = load_class_mapping(json_path)
    print(f"Loaded {len(class_dict)} classes from mapping file.")

    master_mask = None

    # Iterate through the specific files provided via CLI
    for file_path in input_files:
        if not os.path.exists(file_path):
            print(f"Warning: File not found -> {file_path}. Skipping.")
            continue

        filename = os.path.basename(file_path)
        print(f"\n--- Scanning: {filename} ---")

        try:
            with mrcfile.open(file_path, permissive=True) as mrc:
                data = mrc.data

                # Initialize the empty master array on the first valid file
                if master_mask is None:
                    master_mask = np.zeros_like(data, dtype=np.int8)
                    print(f"Initialized master volume with" +
                          f"shape: {master_mask.shape}")

                # Safety check: ensure all tomograms are the same dimensions
                if data.shape != master_mask.shape:
                    print(f"ERROR: {filename} shape {data.shape}" +
                          f"does not match {master_mask.shape}. Skipping.")
                    continue

                # Find unique integers in this specific file
                unique_vals = np.unique(data)

                # Filter out 0 (assuming it's always background)
                labels_to_map = [v for v in unique_vals if v != 0]

                if not labels_to_map:
                    print("Only background (0) found. Skipping to next file.")
                    continue

                # Interactive mapping loop
                for val in labels_to_map:
                    mapped_id = None
                    while mapped_id is None:
                        prompt = f"Found label '{val}' in {filename}."
                        prompt += "What class is this? (Enter ID or name):"
                        user_input = input(prompt).strip().lower()

                        if user_input in lookup:
                            mapped_id = lookup[user_input]
                            print(f"  -> Mapped original label {val} to ID" +
                                  f"{mapped_id}.")
                        else:
                            print("  -> Invalid input. Please enter a valid" +
                                  "ID or class name from your JSON.")

                    # Apply the mapped ID to the master array
                    master_mask[data == val] = mapped_id

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # Save the final compliled array
    if master_mask is not None:
        print(f"\nSaving collated mask to {output_filepath}...")
        with mrcfile.new(output_filepath, overwrite=True) as mrc:
            mrc.set_data(master_mask)
        print("Done! Ready for the 2D slicing script.")
    else:
        print('\nNo valid files were processed Master mask was not created.')


def main():
    msg = "Collate multiple 3D .mrc/.rec annotation files "
    msg += "into a master mapped volume."
    parser = argparse.ArgumentParser(
        description=msg
    )

    msg = "List of input .mrc or .rec files to collate (separated by spaces)."
    parser.add_argument(
        "-i", "--input",
        nargs='+',
        required=True,
        help=msg
    )

    msg = "Path to JSON class mapping file "
    msg += "(default: cryoET_class_mapping.json)."
    parser.add_argument(
        "-j", "--json",
        default="cryoET_class_mapping.json",
        help=msg
    )

    msg = "Output filename for the compiled .mrc file "
    msg += "(default: master_compiled_labels.mrc)."
    parser.add_argument(
        "-o", "--output",
        default="master_compiled_labels.mrc",
        help=msg
    )

    args = parser.parse_args()

    collate_annotations(args.input, args.json, args.output)


if __name__ == "__main__":
    main()
