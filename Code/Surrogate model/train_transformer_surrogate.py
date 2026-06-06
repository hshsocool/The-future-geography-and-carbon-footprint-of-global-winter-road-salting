"""
Train a Transformer surrogate model for pavement temperature prediction.

This script trains a multi-output Transformer model to predict pavement surface
and asphalt-layer temperatures from weather, pavement-depth and time features.

Expected input
--------------
A folder containing CSV files with columns:
- datetime
- relhum_percent
- glohorrad_Whm2
- windspd_ms
- drybulb_C
- D0, D1, D2
- Country_Cluster
- Surface temperature
- Layer1 temperature
- Layer2 temperature
- Layer3 temperature

Outputs
-------
- trained model (.keras)
- feature scaler (.pkl)
- target scaler (.pkl)
- label encoder (.pkl)
- metrics CSV
- training history CSV
- prediction comparison figures

Example
-------
python train_transformer_surrogate.py \
    --input-dir outputs/physics_targets \
    --output-dir outputs/surrogate_model \
    --time-steps 24 \
    --epochs 20 \
    --batch-size 32
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    Add,
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    LayerNormalization,
    MaxPooling1D,
    MultiHeadAttention,
)
from tensorflow.keras.models import Model


FEATURE_COLUMNS = [
    "relhum_percent",
    "glohorrad_Whm2",
    "windspd_ms",
    "drybulb_C",
    "D0",
    "D1",
    "D2",
    "day",
    "weekday",
    "month_sin",
    "month_cos",
    "hour_sin",
    "hour_cos",
    "Country_Cluster_encoded",
]

TARGET_COLUMNS = [
    "Surface temperature",
    "Layer1 temperature",
    "Layer2 temperature",
    "Layer3 temperature",
]


def set_random_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce memory usage by downcasting numeric columns."""
    df = df.copy()

    float_cols = df.select_dtypes(include=["float64"]).columns
    df[float_cols] = df[float_cols].round(4).astype(np.float32)

    int_cols = df.select_dtypes(include=["int64"]).columns
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    return df


def load_training_data(input_dir: Path) -> pd.DataFrame:
    """Load and concatenate all CSV files recursively."""
    csv_files = sorted(input_dir.rglob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    frames = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        frames.append(optimize_dataframe(df))

    combined = pd.concat(frames, ignore_index=True)
    return combined


def add_time_and_categorical_features(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder]:
    """Add cyclical time features and encode Country_Cluster."""
    df = df.copy()

    if "datetime" not in df.columns:
        raise ValueError("'datetime' column is required.")

    if "Country_Cluster" not in df.columns:
        raise ValueError("'Country_Cluster' column is required.")

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).copy()

    df["day"] = df["datetime"].dt.day
    df["weekday"] = df["datetime"].dt.weekday

    df["month_sin"] = np.sin(2 * np.pi * df["datetime"].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["datetime"].dt.month / 12)
    df["hour_sin"] = np.sin(2 * np.pi * df["datetime"].dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["datetime"].dt.hour / 24)

    df["Country_Cluster"] = df["Country_Cluster"].astype("category")

    label_encoder = LabelEncoder()
    df["Country_Cluster_encoded"] = label_encoder.fit_transform(df["Country_Cluster"])

    return df, label_encoder


def validate_model_columns(df: pd.DataFrame) -> None:
    """Validate that all feature and target columns are present."""
    required = FEATURE_COLUMNS + TARGET_COLUMNS
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def scale_and_split(
    df: pd.DataFrame,
    train_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler]:
    """Scale features/targets and split data sequentially."""
    validate_model_columns(df)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMNS]

    scaler_X = MinMaxScaler(feature_range=(-1, 1))
    scaler_y = MinMaxScaler(feature_range=(-1, 1))

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    split_index = int(len(X_scaled) * train_fraction)

    X_train = X_scaled[:split_index]
    X_test = X_scaled[split_index:]
    y_train = y_scaled[:split_index]
    y_test = y_scaled[split_index:]

    return X_train, X_test, y_train, y_test, scaler_X, scaler_y


def create_transformer_sequences(
    X: np.ndarray,
    y: np.ndarray,
    time_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create rolling-window input sequences for Transformer training."""
    X_seq = []
    y_seq = []

    for i in range(len(X) - time_steps):
        X_seq.append(X[i : i + time_steps])
        y_seq.append(y[i + time_steps])

    return np.asarray(X_seq), np.asarray(y_seq)


class PositionalEncoding(layers.Layer):
    """Sinusoidal positional encoding layer."""

    def __init__(self, maxlen: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.maxlen = maxlen
        self.d_model = d_model
        self.pos_encoding = self._positional_encoding(maxlen, d_model)

    @staticmethod
    def _get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def _positional_encoding(self, position, d_model):
        angle_rads = self._get_angles(
            np.arange(position)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model,
        )

        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        return tf.cast(angle_rads[np.newaxis, ...], dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, : tf.shape(inputs)[1], :]

    def get_config(self):
        config = super().get_config()
        config.update({"maxlen": self.maxlen, "d_model": self.d_model})
        return config


def transformer_encoder(
    inputs,
    head_size: int,
    num_heads: int,
    ff_dim: int,
    dropout: float = 0.1,
):
    """Transformer encoder block."""
    x = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=head_size,
        dropout=dropout,
    )(inputs, inputs)
    x = Dropout(dropout)(x)
    x = Add()([x, inputs])
    x = LayerNormalization(epsilon=1e-6)(x)

    x_ff = Dense(ff_dim, activation="relu")(x)
    x_ff = Dropout(dropout)(x_ff)
    x_ff = Dense(inputs.shape[-1])(x_ff)
    x_ff = Add()([x_ff, x])

    return LayerNormalization(epsilon=1e-6)(x_ff)


def build_transformer_model(
    input_shape: tuple[int, int],
    output_dim: int,
    d_model: int = 16,
    num_heads: int = 4,
    ff_dim: int = 384,
    dropout: float = 0.15323787228995225,
    learning_rate: float = 0.00026346160271340934,
    num_encoder_layers: int = 2,
) -> Model:
    """Build and compile the Transformer surrogate model."""
    inputs = Input(shape=input_shape)

    x = Conv1D(
        filters=d_model,
        kernel_size=3,
        padding="same",
        activation="relu",
    )(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dense(d_model)(x)

    # After pooling, the sequence length is reduced.
    pooled_steps = x.shape[1]
    x = PositionalEncoding(maxlen=int(pooled_steps), d_model=d_model)(x)

    for _ in range(num_encoder_layers):
        x = transformer_encoder(
            x,
            head_size=d_model,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
        )

    x = GlobalAveragePooling1D()(x)
    outputs = Dense(output_dim)(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mean_squared_error",
    )

    return model


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_columns: list[str],
) -> pd.DataFrame:
    """Calculate R2, MAE, MSE and RMSE for each target."""
    records = []

    for i, target in enumerate(target_columns):
        mse = mean_squared_error(y_true[:, i], y_pred[:, i])
        records.append(
            {
                "target": target,
                "r2": r2_score(y_true[:, i], y_pred[:, i]),
                "mae": mean_absolute_error(y_true[:, i], y_pred[:, i]),
                "mse": mse,
                "rmse": np.sqrt(mse),
                "observed_mean": float(np.mean(y_true[:, i])),
                "predicted_mean": float(np.mean(y_pred[:, i])),
            }
        )

    return pd.DataFrame(records)


def plot_prediction_comparison(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_columns: list[str],
    output_dir: Path,
    max_points: int = 2000,
) -> None:
    """Save observed-vs-predicted line plots for each target."""
    output_dir.mkdir(parents=True, exist_ok=True)

    n = min(len(y_true), max_points)

    for i, target in enumerate(target_columns):
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(y_true[:n, i], label=f"Observed {target}", linestyle="--")
        ax.plot(y_pred[:n, i], label=f"Predicted {target}")
        ax.set_title(f"Observed vs predicted {target}")
        ax.set_xlabel("Time step")
        ax.set_ylabel(target)
        ax.legend()

        output_file = output_dir / f"prediction_{target.replace(' ', '_').lower()}.png"
        fig.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_training_history(history, output_file: Path) -> None:
    """Save training and validation loss curves."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(history.history["loss"], label="Training loss")
    ax.plot(history.history["val_loss"], label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.legend()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def train_surrogate(args: argparse.Namespace) -> None:
    """Run the complete surrogate model training workflow."""
    set_random_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Loading training data from %s", args.input_dir)
    df = load_training_data(args.input_dir)

    logging.info("Adding time and categorical features")
    df, label_encoder = add_time_and_categorical_features(df)

    X_train, X_test, y_train, y_test, scaler_X, scaler_y = scale_and_split(
        df,
        train_fraction=args.train_fraction,
    )

    logging.info("Creating Transformer sequences with time_steps=%d", args.time_steps)
    X_train_seq, y_train_seq = create_transformer_sequences(
        X_train,
        y_train,
        args.time_steps,
    )
    X_test_seq, y_test_seq = create_transformer_sequences(
        X_test,
        y_test,
        args.time_steps,
    )

    logging.info("Training samples: %s; test samples: %s", X_train_seq.shape, X_test_seq.shape)

    model = build_transformer_model(
        input_shape=(args.time_steps, X_train_seq.shape[2]),
        output_dim=y_train_seq.shape[-1],
        d_model=args.d_model,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        num_encoder_layers=args.num_encoder_layers,
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        X_train_seq,
        y_train_seq,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        callbacks=callbacks,
        verbose=1,
    )

    test_loss = model.evaluate(X_test_seq, y_test_seq, verbose=0)
    logging.info("Test loss: %.6f", test_loss)

    y_pred_scaled = model.predict(X_test_seq)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = scaler_y.inverse_transform(y_test_seq)

    metrics = evaluate_predictions(y_true, y_pred, TARGET_COLUMNS)
    metrics.loc[len(metrics)] = {
        "target": "all_targets",
        "r2": np.nan,
        "mae": np.nan,
        "mse": float(test_loss),
        "rmse": float(np.sqrt(test_loss)),
        "observed_mean": np.nan,
        "predicted_mean": np.nan,
    }

    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    pd.DataFrame(history.history).to_csv(args.output_dir / "training_history.csv", index=False)

    model.save(args.output_dir / "transformer_surrogate.keras")
    joblib.dump(scaler_X, args.output_dir / "scaler_X.pkl")
    joblib.dump(scaler_y, args.output_dir / "scaler_y.pkl")
    joblib.dump(label_encoder, args.output_dir / "label_encoder.pkl")

    mapping = dict(
        zip(
            label_encoder.classes_.tolist(),
            label_encoder.transform(label_encoder.classes_).tolist(),
        )
    )
    with (args.output_dir / "country_cluster_mapping.json").open("w", encoding="utf-8") as file:
        json.dump(mapping, file, indent=2, ensure_ascii=False)

    plot_training_history(history, args.output_dir / "training_loss.png")
    plot_prediction_comparison(
        y_true,
        y_pred,
        TARGET_COLUMNS,
        args.output_dir / "prediction_plots",
        max_points=args.max_plot_points,
    )

    logging.info("Saved model and outputs to %s", args.output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Transformer surrogate model for pavement temperature prediction."
    )

    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)

    parser.add_argument("--time-steps", default=24, type=int)
    parser.add_argument("--train-fraction", default=0.9, type=float)
    parser.add_argument("--validation-split", default=0.1, type=float)

    parser.add_argument("--d-model", default=16, type=int)
    parser.add_argument("--num-heads", default=4, type=int)
    parser.add_argument("--ff-dim", default=384, type=int)
    parser.add_argument("--dropout", default=0.15323787228995225, type=float)
    parser.add_argument("--learning-rate", default=0.00026346160271340934, type=float)
    parser.add_argument("--num-encoder-layers", default=2, type=int)

    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--patience", default=10, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--max-plot-points", default=2000, type=int)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    train_surrogate(args)


if __name__ == "__main__":
    main()
