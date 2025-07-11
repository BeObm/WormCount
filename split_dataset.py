#!/usr/bin/env python3
"""
Example script showing how to use the dataset splitter
"""

from split_dataset_utils import split_dataset, analyze_dataset


def main():
    """
    Example usage of the dataset splitter
    """

    # Define your paths
    images_directory = "dataset/images/train"  # Replace with your images folder
    labels_directory = "dataset/labels/train"  # Replace with your labels folder
    output_directory = "worm_dataset"  # Where to save the split dataset

    print("Worm Detection Dataset Splitter")
    print("=" * 40)

    # Step 1: Analyze the dataset first (optional but recommended)
    print("Step 1: Analyzing dataset...")
    analyze_dataset(images_directory, labels_directory)

    # Step 2: Split the dataset
    print("\nStep 2: Splitting dataset...")
    try:
        output_path, summary = split_dataset(
            images_dir=images_directory,
            labels_dir=labels_directory,
            output_dir=output_directory,
            train_ratio=0.7,  # 70% for training
            val_ratio=0.2,  # 20% for validation
            test_ratio=0.1,  # 10% for testing
            validate_labels=True,
            random_seed=42
        )

        print(f"\nDataset split completed!")
        print(f"Dataset structure created at: {output_path}")

        # Print summary
        print(f"\nSummary:")
        print(f"  Total images: {summary['total_images']}")
        print(f"  Training samples: {summary['splits']['train']['count']}")
        print(f"  Validation samples: {summary['splits']['val']['count']}")
        print(f"  Test samples: {summary['splits']['test']['count']}")

        if summary['missing_labels']:
            print(f"  Missing labels: {summary['missing_labels']}")

    except Exception as e:
        print(f"Error: {e}")
        print("Please check your file paths and try again.")


def example_with_different_ratios():
    """
    Example with different train/val/test ratios
    """

    images_directory = "path/to/your/images"
    labels_directory = "path/to/your/labels"
    output_directory = "dataset_split_80_15_5"

    # Different ratio: 80% train, 15% val, 5% test
    split_dataset(
        images_dir=images_directory,
        labels_dir=labels_directory,
        output_dir=output_directory,
        train_ratio=0.8,
        val_ratio=0.15,
        test_ratio=0.05,
        random_seed=123  # Different seed for different split
    )


def example_no_test_set():
    """
    Example with only train/val split (no test set)
    """

    images_directory = "path/to/your/images"
    labels_directory = "path/to/your/labels"
    output_directory = "dataset_split_train_val_only"

    # Only train and validation sets
    split_dataset(
        images_dir=images_directory,
        labels_dir=labels_directory,
        output_dir=output_directory,
        train_ratio=0.8,
        val_ratio=0.2,
        test_ratio=0.0,  # No test set
        random_seed=42
    )


if __name__ == "__main__":
    # Run the main example
    main()

    # Uncomment to run other examples:
    # example_with_different_ratios()
    # example_no_test_set()