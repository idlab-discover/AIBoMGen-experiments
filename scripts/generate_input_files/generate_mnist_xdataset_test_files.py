import os
import numpy as np
import tensorflow as tf
from PIL import Image
import yaml
import shutil


def generate_mnist_xdataset_test_files(output_dir, dataset_sizes, max_per_class=500):
    """
    Generates multiple datasets of increasing size, corresponding YAML definitions, and a single model for MNIST testing.

    Args:
        output_dir (str): The directory where the generated files will be saved.
        dataset_sizes (list): List of dataset sizes (number of images per class) to generate.
        max_per_class (int): Maximum number of images to save per class.

    Returns:
        dict: Paths to the generated files (datasets, YAML definitions, and model).
    """
    os.makedirs(output_dir, exist_ok=True)

    # File paths
    model_filename = os.path.join(output_dir, "mnist_model.keras")
    dataset_files = []
    yaml_files = []

    # Load MNIST data
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype(np.float32) / 255.0  # Normalize to [0, 1]
    x_train = np.expand_dims(x_train, -1)  # Add channel dimension

    # Generate datasets of increasing size
    for size in dataset_sizes:
        dataset_folder = os.path.join(output_dir, f"mnist_dataset_{size}")
        yaml_filename = os.path.join(
            output_dir, f"mnist_definition_{size}.yaml")
        zip_filename = os.path.join(output_dir, f"mnist_dataset_{size}.zip")

        os.makedirs(dataset_folder, exist_ok=True)
        class_counts = {i: 0 for i in range(10)}
        for class_idx in range(10):
            class_folder = os.path.join(dataset_folder, f"class_{class_idx}")
            os.makedirs(class_folder, exist_ok=True)
        for idx, (img, label) in enumerate(zip(x_train, y_train)):
            if class_counts[label] >= size:
                continue
            img = (img * 255).astype(np.uint8)
            img_pil = Image.fromarray(img.squeeze(), mode="L")
            img_path = os.path.join(
                dataset_folder, f"class_{label}", f"image_{class_counts[label]}.png")
            img_pil.save(img_path)
            class_counts[label] += 1
            # Stop early if all classes are filled
            if all(count >= size for count in class_counts.values()):
                break
        print(
            f"MNIST dataset with {size} images per class saved to {dataset_folder}")

        # Save YAML definition
        dataset_definition = {
            "type": "image",
            "image_size": [28, 28],
            "input_shape": [28, 28, 3],
            "output_shape": [10],
            "preprocessing": {
                "normalize": True
            },
            "dataset_size_per_class": size
        }
        with open(yaml_filename, "w") as yaml_file:
            yaml.dump(dataset_definition, yaml_file)
        print(
            f"Dataset definition for {size} images per class saved to {yaml_filename}")

        # Zip the dataset
        shutil.make_archive(base_name=zip_filename.replace(
            ".zip", ""), format="zip", root_dir=dataset_folder)
        print(f"Dataset zipped to {zip_filename}")

        dataset_files.append(zip_filename)
        yaml_files.append(yaml_filename)

    # Create and save a single model
    def create_mnist_model():
        """Create a simple CNN model for MNIST."""
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
        "datasets": dataset_files,
        "yamls": yaml_files,
        "model": model_filename
    }


# Example usage (optional, for testing purposes)
if __name__ == "__main__":
    root_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "../.."))
    tmp_folder = os.path.join(
        root_dir, "results", "generate_mnist_xdataset_test_files")
    dataset_sizes = [100, 200, 300, 400, 500]  # Number of images per class
    generate_mnist_xdataset_test_files(tmp_folder, dataset_sizes)
