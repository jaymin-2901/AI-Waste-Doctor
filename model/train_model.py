"""Train the waste classifier from real labeled images.

Supported dataset layouts:
    dataset/
        Recyclable/
        Dry Waste/
        Wet Waste/

or the Kaggle layout:
    dataset/
        Recyclable/
        Non-Recyclable/
        Organic/

Optional e-waste examples can be added inside `Recyclable/E-Waste/`.
They are learned as the app's `Recyclable` category.

Each directory should contain photos of objects belonging to that class.
"""

import argparse
import os
from pathlib import Path

# Keep TensorFlow on CPU on native Windows and suppress startup noise.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.get_logger().setLevel("ERROR")

CLASS_NAMES = ["Recyclable", "Dry Waste", "Wet Waste"]
KAGGLE_CLASS_NAMES = ["Recyclable", "Non-Recyclable", "Organic"]
IMAGE_SIZE = (224, 224)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def _count_images(directory: Path) -> int:
    return sum(
        1
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def train(
    dataset_dir: Path,
    output_path: Path,
    epochs: int,
    batch_size: int,
    fine_tune_epochs: int = 8,
    fine_tune_layers: int = 30,
) -> None:
    """Train and save a MobileNetV2 waste classifier."""
    if all((dataset_dir / name).is_dir() for name in CLASS_NAMES):
        source_class_names = CLASS_NAMES
        source_to_app_index = [0, 1, 2]
    elif all((dataset_dir / name).is_dir() for name in KAGGLE_CLASS_NAMES):
        source_class_names = KAGGLE_CLASS_NAMES
        source_to_app_index = [0, 1, 2]
    else:
        raise FileNotFoundError(
            f"Expected either {CLASS_NAMES} or {KAGGLE_CLASS_NAMES} folders under {dataset_dir}."
        )

    class_counts = {
        class_name: _count_images(dataset_dir / class_name)
        for class_name in source_class_names
    }
    e_waste_images = sum(
        _count_images(path)
        for path in (dataset_dir / "Recyclable").rglob("*")
        if path.is_dir() and path.name.lower().replace("_", "-") == "e-waste"
    )
    print(f"Dataset image counts: {class_counts}")
    if e_waste_images == 0:
        print(
            "WARNING: No e-waste images found. Add phone/electronics photos to "
            f"{dataset_dir / 'Recyclable'} before expecting reliable mobile-device predictions."
        )
    else:
        print(f"Found {e_waste_images} e-waste images under the Recyclable class.")

    train_ds = keras.utils.image_dataset_from_directory(
        dataset_dir,
        class_names=source_class_names,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        validation_split=0.2,
        subset="training",
        seed=42,
    )
    validation_ds = keras.utils.image_dataset_from_directory(
        dataset_dir,
        class_names=source_class_names,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        validation_split=0.2,
        subset="validation",
        seed=42,
    )

    preprocess = keras.applications.mobilenet_v2.preprocess_input
    train_ds = train_ds.map(lambda images, labels: (preprocess(images), labels))
    validation_ds = validation_ds.map(lambda images, labels: (preprocess(images), labels))

    # Keep the app's stable class order even when using Kaggle folder names.
    class_index_map = tf.constant(source_to_app_index, dtype=tf.int32)
    train_ds = train_ds.map(lambda images, labels: (images, tf.gather(class_index_map, labels)))
    validation_ds = validation_ds.map(lambda images, labels: (images, tf.gather(class_index_map, labels)))
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    validation_ds = validation_ds.prefetch(tf.data.AUTOTUNE)

    total_images = sum(class_counts.values())
    class_weights = {
        index: total_images / (len(source_class_names) * class_counts[class_name])
        for index, class_name in enumerate(source_class_names)
        if class_counts[class_name]
    }

    augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.15),
            layers.RandomContrast(0.15),
        ],
        name="augmentation",
    )

    base_model = keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(*IMAGE_SIZE, 3))
    x = augmentation(inputs)
    x = base_model(x, training=False)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(len(CLASS_NAMES), activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=2, min_lr=1e-6
        ),
    ]
    model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        shuffle=False,
    )

    if fine_tune_epochs > 0 and fine_tune_layers > 0:
        base_model.trainable = True
        for layer in base_model.layers[:-fine_tune_layers]:
            layer.trainable = False
        for layer in base_model.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(
            train_ds,
            validation_data=validation_ds,
            epochs=fine_tune_epochs,
            callbacks=callbacks,
            class_weight=class_weights,
            shuffle=False,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    labels_path = output_path.with_name("labels.txt")
    labels_path.write_text(
        "\n".join(f"{index} {name}" for index, name in enumerate(CLASS_NAMES)) + "\n",
        encoding="utf-8",
    )
    print(f"Saved model to {output_path}")
    print(f"Saved labels to {labels_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument(
        "--output", type=Path, default=Path("model") / "keras_model.h5"
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--fine-tune-epochs", type=int, default=8)
    parser.add_argument("--fine-tune-layers", type=int, default=30)
    args = parser.parse_args()
    train(
        args.dataset,
        args.output,
        args.epochs,
        args.batch_size,
        args.fine_tune_epochs,
        args.fine_tune_layers,
    )


if __name__ == "__main__":
    main()
