import os
import numpy as np
import tensorflow as tf
from PIL import Image
import yaml
import shutil


def generate_mnist_test_files(output_dir, max_per_class=500):
    """
    Generates a dataset, YAML definition, and a simple model for MNIST testing.

    Args:
        output_dir (str): The directory where the generated files will be saved.
        max_per_class (int): Maximum number of images to save per class.

    Returns:
        dict: Paths to the generated files (dataset, YAML, and model).
    """
    os.makedirs(output_dir, exist_ok=True)

    # File paths
    dataset_folder = os.path.join(output_dir, "mnist_dataset")
    yaml_filename = os.path.join(output_dir, "mnist_definition.yaml")
    model_filename = os.path.join(output_dir, "mnist_model.keras")
    zip_filename = os.path.join(output_dir, "mnist_dataset.zip")

    # Load MNIST data
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype(np.float32) / 255.0  # Normalize to [0, 1]
    x_train = np.expand_dims(x_train, -1)  # Add channel dimension

    # Save images to disk
    os.makedirs(dataset_folder, exist_ok=True)
    class_counts = {i: 0 for i in range(10)}
    for class_idx in range(10):
        class_folder = os.path.join(dataset_folder, f"class_{class_idx}")
        os.makedirs(class_folder, exist_ok=True)
    for idx, (img, label) in enumerate(zip(x_train, y_train)):
        if class_counts[label] >= max_per_class:
            continue
        img = (img * 255).astype(np.uint8)
        img_pil = Image.fromarray(img.squeeze(), mode="L")
        img_path = os.path.join(
            dataset_folder, f"class_{label}", f"image_{class_counts[label]}.png")
        img_pil.save(img_path)
        class_counts[label] += 1
        # Stop early if all classes are filled
        if all(count >= max_per_class for count in class_counts.values()):
            break
    print(
        f"MNIST images saved to {dataset_folder} (max {max_per_class} per class)")

    # Save YAML definition
    dataset_definition = {
        "type": "image",
        "image_size": [28, 28],
        "input_shape": [28, 28, 3],
        "output_shape": [10],
        "preprocessing": {
            "normalize": True
        }
    }
    with open(yaml_filename, "w") as yaml_file:
        yaml.dump(dataset_definition, yaml_file)
    print(f"Dataset definition saved to {yaml_filename}")

    # Zip the dataset
    shutil.make_archive(base_name=zip_filename.replace(
        ".zip", ""), format="zip", root_dir=dataset_folder)
    print(f"Dataset zipped to {zip_filename}")

    # Create and save a simple CNN model
    def create_mnist_model():
        """Create and train a simple CNN model for MNIST."""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(28, 28, 3)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    model = create_mnist_model()
    model.save(model_filename)
    print(f"Model saved to {model_filename}")

    return {
        "dataset": zip_filename,
        "yaml": yaml_filename,
        "model": model_filename
    }


# Example usage (optional, for testing purposes)
if __name__ == "__main__":
    root_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "../.."))
    tmp_folder = os.path.join(root_dir, "results", "generate_mnist_test_files")
    generate_mnist_test_files(tmp_folder)
