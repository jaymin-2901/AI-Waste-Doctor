"""Evaluate a waste model on a deterministic held-out validation split."""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

CLASS_NAMES = ["Recyclable", "Non-Recyclable", "Organic"]
APP_LABELS = ["Recyclable", "Dry Waste", "Wet Waste"]
IMAGE_SIZE = (224, 224)


def evaluate(dataset_dir: Path, model_path: Path, batch_size: int) -> None:
    dataset = keras.utils.image_dataset_from_directory(
        dataset_dir,
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        validation_split=0.2,
        subset="validation",
        seed=42,
        shuffle=True,
    )
    print(f"source_classes={dataset.class_names}")
    model = keras.models.load_model(model_path, compile=False)
    predictions = []
    actual = []
    for images, labels in dataset:
        images = keras.applications.mobilenet_v2.preprocess_input(tf.cast(images, tf.float32))
        predictions.extend(np.argmax(model(images, training=False).numpy(), axis=1))
        actual.extend(labels.numpy())

    matrix = np.zeros((len(APP_LABELS), len(APP_LABELS)), dtype=np.int64)
    for expected, predicted in zip(actual, predictions):
        matrix[int(expected), int(predicted)] += 1

    print("Held-out validation metrics")
    print("class,samples,precision,recall,f1")
    for index, label in enumerate(APP_LABELS):
        true_positive = matrix[index, index]
        predicted_positive = matrix[:, index].sum()
        actual_positive = matrix[index, :].sum()
        precision = true_positive / predicted_positive if predicted_positive else 0.0
        recall = true_positive / actual_positive if actual_positive else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        print(f"{label},{actual_positive},{precision:.4f},{recall:.4f},{f1:.4f}")

    accuracy = np.trace(matrix) / matrix.sum() if matrix.sum() else 0.0
    print(f"overall_accuracy,{accuracy:.4f}")
    print("confusion_matrix_rows=actual,columns=predicted")
    print(matrix)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--model", type=Path, default=Path("model") / "keras_model.h5")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    evaluate(args.dataset, args.model, args.batch_size)


if __name__ == "__main__":
    main()