"""
train.py — Load collected landmark data and train the gesture classifier.

Usage:
    python train.py

Outputs:
    model/gesture_model.keras    trained model
    model/training_report.txt    val accuracy + per-class report
"""

import os
import sys
import json
import numpy as np


# ── Config ────────────────────────────────────────────────────────────────────

def _load_config() -> tuple[dict, dict]:
    with open("config/app_config.json",     "r") as f:
        app = json.load(f)
    with open("config/gesture_config.json", "r") as f:
        ges = json.load(f)
    return app, ges


# ── Data loading ──────────────────────────────────────────────────────────────

def load_dataset(gesture_names: list[str],
                 data_dir: str) -> tuple[np.ndarray, np.ndarray]:
    X, y    = [], []
    missing = []

    for idx, name in enumerate(gesture_names):
        path = os.path.join(data_dir, f"{name}.npy")
        if not os.path.exists(path):
            missing.append(name)
            continue
        arr = np.load(path)
        X.append(arr)
        y.extend([idx] * len(arr))
        print(f"  {name:>8}  {len(arr):>4} samples")

    if missing:
        print(f"\n[WARNING] No data for: {', '.join(missing)}")
        print("Run collector/app.py to collect missing gestures.\n")

    if not X:
        print("[ERROR] No data found. Aborting.")
        sys.exit(1)

    return np.vstack(X), np.array(y)


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(num_classes: int):
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(63,)),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.35),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.30),
        tf.keras.layers.Dense(64,  activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.20),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── Entry ─────────────────────────────────────────────────────────────────────

def main() -> None:
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from sklearn.metrics         import classification_report

    app_cfg, ges_cfg = _load_config()
    data_dir  = app_cfg["paths"]["data_dir"]
    model_dir = app_cfg["paths"]["model_dir"]

    from core.gesture_manager import GestureManager
    gm    = GestureManager("config/gesture_config.json")
    names = gm.gesture_names

    print(f"\nHandSpeak — Training on {len(names)} gestures")
    print(f"Labels: {', '.join(names)}\n")
    print(f"Loading data from '{data_dir}/' ...\n")

    X, y = load_dataset(names, data_dir)
    print(f"\nTotal samples: {len(X)}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)}   Val: {len(X_val)}\n")

    model = build_model(gm.num_classes)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=10,
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, verbose=1
        ),
    ]

    print("\nTraining ...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
    )

    loss, acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nFinal Val Accuracy: {acc * 100:.2f}%")

    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    report = classification_report(y_val, y_pred, target_names=names)
    print("\nClassification Report:\n")
    print(report)

    os.makedirs(model_dir, exist_ok=True)
    model_path  = os.path.join(model_dir, "gesture_model.keras")
    report_path = os.path.join(model_dir, "training_report.txt")

    model.save(model_path)
    with open(report_path, "w") as f:
        f.write(f"Val Accuracy: {acc * 100:.2f}%\n\n")
        f.write(report)

    print(f"\nModel  → {model_path}")
    print(f"Report → {report_path}")
    print("\nDone. Run main.py to launch HandSpeak.")


if __name__ == "__main__":
    main()