import os
import numpy as np
import tensorflow as tf
from PIL import Image
import yaml
import shutil


def generate_mnist_xmodel_test_files(output_dir, conv_units_list, max_per_class=500):
    """
    Generates a dataset, YAML definition, and multiple models for MNIST testing.

    Args:
        output_dir (str): The directory where the generated files will be saved.
        conv_units_list (list): List of convolutional unit sizes for the models.
        max_per_class (int): Maximum number of images to save per class.

    Returns:
        dict: Paths to the generated files (dataset, YAML, and models).
    """
    os.makedirs(output_dir, exist_ok=True)

    # File paths
    dataset_folder = os.path.join(output_dir, "mnist_dataset")
    yaml_filename = os.path.join(output_dir, "mnist_definition.yaml")
    zip_filename = os.path.join(output_dir, "mnist_dataset.zip")
    model_files = []

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

    # Create and save models with varying convolutional units
    def create_mnist_model(conv_units):
        """Create a simple CNN model for MNIST."""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(28, 28, 3)),
            tf.keras.layers.Conv2D(conv_units, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    for conv_units in conv_units_list:
        model_filename = os.path.join(
            output_dir, f"mnist_model_{conv_units}_units.keras")
        model = create_mnist_model(conv_units)
        model.save(model_filename)
        model_files.append(model_filename)
        print(f"Model with {conv_units} units saved to {model_filename}")

    return {
        "dataset": zip_filename,
        "yaml": yaml_filename,
        "models": model_files
    }


# Example usage (optional, for testing purposes)
if __name__ == "__main__":
    root_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "../.."))
    tmp_folder = os.path.join(
        root_dir, "results", "generate_mnist_xmodels_test_files")
    conv_units_list = [8, 16, 32, 64, 128]
    generate_mnist_xmodel_test_files(tmp_folder, conv_units_list)
