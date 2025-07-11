import os
import shutil
import random
from pathlib import Path
import argparse
from tqdm import tqdm
import json


def create_dataset_structure(output_dir):
    """
    Create the dataset directory structure

    output_dir/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
    """
    output_path = Path(output_dir)

    # Create all required directories
    directories = [
        output_path / "train" / "images",
        output_path / "train" / "labels",
        output_path / "val" / "images",
        output_path / "val" / "labels",
        output_path / "test" / "images",
        output_path / "test" / "labels"
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

    return output_path


def find_image_label_pairs(images_dir, labels_dir):
    """
    Find matching image-label pairs from separate directories
    """
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)

    # Supported image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']

    # Get all image files
    image_files = []
    for ext in image_extensions:
        image_files.extend(images_path.glob(f"*{ext}"))
        image_files.extend(images_path.glob(f"*{ext.upper()}"))

    print(f"Found {len(image_files)} image files")

    # Find corresponding label files
    valid_pairs = []
    missing_labels = []

    for img_file in image_files:
        # Look for corresponding label file
        label_file = labels_path / f"{img_file.stem}.txt"

        if label_file.exists():
            valid_pairs.append((img_file, label_file))
        else:
            missing_labels.append(img_file.name)

    # Report results
    print(f"Found {len(valid_pairs)} valid image-label pairs")

    if missing_labels:
        print(f"Warning: {len(missing_labels)} images have no corresponding labels:")
        for i, missing in enumerate(missing_labels[:10]):  # Show first 10
            print(f"  - {missing}")
        if len(missing_labels) > 10:
            print(f"  ... and {len(missing_labels) - 10} more")

    return valid_pairs, missing_labels


def validate_label_file(label_file):
    """
    Validate YOLO format label file
    """
    try:
        with open(label_file, 'r') as f:
            lines = f.readlines()

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue

            parts = line.split()
            if len(parts) < 5:
                return False, f"Line {line_num}: Expected 5 values, got {len(parts)}"

            try:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                # Check ranges (YOLO format should be normalized 0-1)
                if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and
                        0 <= width <= 1 and 0 <= height <= 1):
                    return False, f"Line {line_num}: Coordinates out of range [0,1]"

                if class_id < 0:
                    return False, f"Line {line_num}: Class ID must be non-negative"

            except ValueError as e:
                return False, f"Line {line_num}: Invalid number format - {e}"

        return True, "Valid"

    except Exception as e:
        return False, f"Error reading file: {e}"


def split_dataset(images_dir, labels_dir, output_dir,
                  train_ratio=0.7, val_ratio=0.2, test_ratio=0.1,
                  validate_labels=True, random_seed=42):
    """
    Split dataset into train/validation/test sets

    Args:
        images_dir: Directory containing all images
        labels_dir: Directory containing all labels
        output_dir: Output directory for split dataset
        train_ratio: Ratio for training set (default: 0.7)
        val_ratio: Ratio for validation set (default: 0.2)
        test_ratio: Ratio for test set (default: 0.1)
        validate_labels: Whether to validate label files (default: True)
        random_seed: Random seed for reproducible splits (default: 42)
    """

    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Train, validation, and test ratios must sum to 1.0")

    # Set random seed for reproducibility
    random.seed(random_seed)

    print("Starting dataset split...")
    print(f"Ratios - Train: {train_ratio:.1%}, Val: {val_ratio:.1%}, Test: {test_ratio:.1%}")
    print(f"Random seed: {random_seed}")

    # Create output directory structure
    output_path = create_dataset_structure(output_dir)

    # Find image-label pairs
    valid_pairs, missing_labels = find_image_label_pairs(images_dir, labels_dir)

    if len(valid_pairs) == 0:
        raise ValueError("No valid image-label pairs found!")

    # Validate label files if requested
    if validate_labels:
        print("\nValidating label files...")
        invalid_labels = []

        for img_file, label_file in tqdm(valid_pairs, desc="Validating labels"):
            is_valid, message = validate_label_file(label_file)
            if not is_valid:
                invalid_labels.append((label_file.name, message))

        if invalid_labels:
            print(f"Warning: {len(invalid_labels)} invalid label files found:")
            for label_name, error in invalid_labels[:5]:  # Show first 5
                print(f"  - {label_name}: {error}")
            if len(invalid_labels) > 5:
                print(f"  ... and {len(invalid_labels) - 5} more")

            # Option to continue or stop
            response = input("Continue with invalid labels? (y/n): ").lower()
            if response != 'y':
                print("Dataset split cancelled.")
                return

    # Shuffle pairs for random distribution
    random.shuffle(valid_pairs)

    # Calculate split sizes
    total_pairs = len(valid_pairs)
    train_size = int(total_pairs * train_ratio)
    val_size = int(total_pairs * val_ratio)
    test_size = total_pairs - train_size - val_size  # Remaining goes to test

    # Split the pairs
    train_pairs = valid_pairs[:train_size]
    val_pairs = valid_pairs[train_size:train_size + val_size]
    test_pairs = valid_pairs[train_size + val_size:]

    print(f"\nDataset split:")
    print(f"  Training: {len(train_pairs)} samples ({len(train_pairs) / total_pairs:.1%})")
    print(f"  Validation: {len(val_pairs)} samples ({len(val_pairs) / total_pairs:.1%})")
    print(f"  Test: {len(test_pairs)} samples ({len(test_pairs) / total_pairs:.1%})")

    # Copy files to respective directories
    def copy_pairs(pairs, split_name):
        print(f"\nCopying {split_name} files...")
        for img_file, label_file in tqdm(pairs, desc=f"Copying {split_name}"):
            # Copy image
            img_dest = output_path / split_name / "images" / img_file.name
            shutil.copy2(img_file, img_dest)

            # Copy label
            label_dest = output_path / split_name / "labels" / label_file.name
            shutil.copy2(label_file, label_dest)

    # Copy files for each split
    copy_pairs(train_pairs, "train")
    copy_pairs(val_pairs, "val")
    copy_pairs(test_pairs, "test")

    # Create summary report
    summary = {
        "total_images": len(valid_pairs),
        "missing_labels": len(missing_labels),
        "splits": {
            "train": {
                "count": len(train_pairs),
                "ratio": len(train_pairs) / total_pairs,
                "files": [p[0].name for p in train_pairs]
            },
            "val": {
                "count": len(val_pairs),
                "ratio": len(val_pairs) / total_pairs,
                "files": [p[0].name for p in val_pairs]
            },
            "test": {
                "count": len(test_pairs),
                "ratio": len(test_pairs) / total_pairs,
                "files": [p[0].name for p in test_pairs]
            }
        },
        "missing_labels": missing_labels,
        "random_seed": random_seed
    }

    # Save summary
    with open(output_path / "split_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nDataset split completed successfully!")
    print(f"Output directory: {output_path}")
    print(f"Summary saved to: {output_path / 'split_summary.json'}")

    return output_path, summary


def analyze_dataset(images_dir, labels_dir):
    """
    Analyze the dataset before splitting
    """
    print("Analyzing dataset...")

    # Find pairs
    valid_pairs, missing_labels = find_image_label_pairs(images_dir, labels_dir)

    if len(valid_pairs) == 0:
        print("No valid pairs found!")
        return

    # Analyze labels
    class_counts = {}
    total_objects = 0
    empty_labels = 0

    print("\nAnalyzing labels...")
    for img_file, label_file in tqdm(valid_pairs, desc="Analyzing"):
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()

            non_empty_lines = [line.strip() for line in lines if line.strip()]

            if not non_empty_lines:
                empty_labels += 1
                continue

            for line in non_empty_lines:
                parts = line.split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    class_counts[class_id] = class_counts.get(class_id, 0) + 1
                    total_objects += 1

        except Exception as e:
            print(f"Error processing {label_file}: {e}")

    # Print analysis
    print(f"\nDataset Analysis:")
    print(f"  Total valid image-label pairs: {len(valid_pairs)}")
    print(f"  Missing labels: {len(missing_labels)}")
    print(f"  Empty label files: {empty_labels}")
    print(f"  Total objects: {total_objects}")
    print(f"  Average objects per image: {total_objects / len(valid_pairs):.2f}")

    if class_counts:
        print(f"\nClass distribution:")
        for class_id, count in sorted(class_counts.items()):
            print(f"  Class {class_id}: {count} objects ({count / total_objects:.1%})")


def main():
    """
    Main function with command line interface
    """
    parser = argparse.ArgumentParser(description='Split image dataset into train/val/test sets')

    parser.add_argument('--images', type=str, required=True,
                        help='Directory containing all images')
    parser.add_argument('--labels', type=str, required=True,
                        help='Directory containing all labels')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for split dataset')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='Training set ratio (default: 0.7)')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                        help='Validation set ratio (default: 0.2)')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                        help='Test set ratio (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible splits (default: 42)')
    parser.add_argument('--no-validate', action='store_true',
                        help='Skip label validation')
    parser.add_argument('--analyze-only', action='store_true',
                        help='Only analyze dataset, do not split')

    args = parser.parse_args()

    # Validate arguments
    if not os.path.exists(args.images):
        print(f"Error: Images directory '{args.images}' does not exist")
        return

    if not os.path.exists(args.labels):
        print(f"Error: Labels directory '{args.labels}' does not exist")
        return

    if args.analyze_only:
        analyze_dataset(args.images, args.labels)
        return

    # Split dataset
    try:
        output_path, summary = split_dataset(
            images_dir=args.images,
            labels_dir=args.labels,
            output_dir=args.output,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            validate_labels=not args.no_validate,
            random_seed=args.seed
        )

        print(f"\nSplit completed successfully!")
        print(f"Use the following directories for training:")
        print(f"  Train: {output_path / 'train'}")
        print(f"  Val: {output_path / 'val'}")
        print(f"  Test: {output_path / 'test'}")

    except Exception as e:
        print(f"Error during split: {e}")


if __name__ == "__main__":
    main()