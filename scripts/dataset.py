from finetune_llms.custom_dataset import load_custom_dataset


def main():
    """Load dataset and print first 10 items."""
    # You can modify this path to point to your data directory
    data_path = "data/eng"  # Change this to your actual data directory

    print(f"Loading dataset from: {data_path}")
    dataset = load_custom_dataset(data_path)

    print("Dataset loaded successfully!")
    print(f"Train split: {len(dataset['train'])} items")
    print(f"Validation split: {len(dataset['validation'])} items")
    print("\n" + "=" * 50)
    print("First 10 items from training set:")
    print("=" * 50 + "\n")

    # Output first 10 items from train split
    for i, item in enumerate(dataset["train"]):
        if i >= 10:
            break

        print(f"Item {i + 1}:")
        print("-" * 20)
        print(item["text"])
        print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    main()
