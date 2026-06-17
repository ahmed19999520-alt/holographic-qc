from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from typing import Dict, List, Optional, Tuple


class VirasoroEquivariantLayer(keras.layers.Layer):
    def __init__(self, units: int, central_charge: float, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.c = central_charge

    def build(self, input_shape):
        self.W = self.add_weight(
            name="W",
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="b",
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
        )
        self.c_weight = self.add_weight(
            name="c_scale",
            shape=(1,),
            initializer=keras.initializers.Constant(self.c),
            trainable=False,
        )
        super().build(input_shape)

    def call(self, inputs):
        linear = tf.matmul(inputs, self.W) + self.b
        scale = self.c_weight / 6.0
        return linear * tf.cast(scale, linear.dtype)

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units, "central_charge": float(self.c)})
        return config


class HolographicDecoherenceNet(keras.Model):
    def __init__(
        self,
        central_charge: float,
        n_materials: int = 3,
        hidden_dims: List[int] = None,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.c = central_charge
        hidden_dims = hidden_dims or [128, 256, 128, 64]

        self.feature_encoder = keras.Sequential([
            layers.Dense(hidden_dims[0], activation="gelu",
                         kernel_regularizer=regularizers.l2(1e-4)),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
        ], name="feature_encoder")

        self.virasoro_layer = VirasoroEquivariantLayer(
            hidden_dims[1], central_charge, name="virasoro"
        )

        self.bulk_propagator_net = keras.Sequential([
            layers.Dense(hidden_dims[2], activation="gelu"),
            layers.Dense(hidden_dims[3], activation="gelu"),
        ], name="bulk_propagator")

        self.decoherence_head = keras.Sequential([
            layers.Dense(32, activation="gelu"),
            layers.Dense(1, activation="softplus"),
        ], name="decoherence_head")

        self.enhancement_head = keras.Sequential([
            layers.Dense(32, activation="gelu"),
            layers.Dense(1, activation="softplus"),
        ], name="enhancement_head")

    def call(self, inputs, training=False):
        x = self.feature_encoder(inputs, training=training)
        x = self.virasoro_layer(x)
        x = tf.nn.gelu(x)
        x = self.bulk_propagator_net(x, training=training)
        decoherence_rate = self.decoherence_head(x)
        enhancement = self.enhancement_head(x)
        return {"decoherence_rate": decoherence_rate, "enhancement_factor": enhancement}

    def predict_coherence_time(self, inputs: np.ndarray) -> np.ndarray:
        outputs = self(tf.cast(inputs, tf.float32), training=False)
        rate = outputs["decoherence_rate"].numpy().flatten()
        return 1.0 / (rate + 1e-30)


class AdsCftCorrelatorNet(keras.Model):
    def __init__(
        self,
        central_charge: float,
        ads_radius: float,
        n_positions: int = 100,
        hidden_dims: List[int] = None,
    ):
        super().__init__()
        self.c = central_charge
        self.L = ads_radius
        hidden_dims = hidden_dims or [64, 128, 64, 32]

        self.position_encoder = keras.Sequential([
            layers.Dense(hidden_dims[0], activation="gelu"),
            layers.Dense(hidden_dims[1], activation="gelu"),
        ], name="position_encoder")

        self.z_encoder = keras.Sequential([
            layers.Dense(hidden_dims[0] // 2, activation="gelu"),
            layers.Dense(hidden_dims[1] // 2, activation="gelu"),
        ], name="z_encoder")

        self.propagator_head = keras.Sequential([
            layers.Dense(hidden_dims[2], activation="gelu"),
            layers.Dense(hidden_dims[3], activation="gelu"),
            layers.Dense(1, activation="softplus"),
        ], name="propagator_head")

    def call(self, inputs, training=False):
        z = inputs[:, :1]
        x = inputs[:, 1:]
        z_feat = self.z_encoder(z, training=training)
        x_feat = self.position_encoder(x, training=training)
        combined = tf.concat([z_feat, x_feat], axis=-1)
        return self.propagator_head(combined, training=training)

    def loss_fn(self, y_true, y_pred):
        return tf.reduce_mean(tf.square(tf.math.log(y_pred + 1e-10) - tf.math.log(y_true + 1e-10)))


class EntanglementEntropyNet(keras.Model):
    def __init__(self, central_charge: float, max_interval: float):
        super().__init__()
        self.c = central_charge
        self.L_max = max_interval

        self.net = keras.Sequential([
            layers.Dense(64, activation="gelu", input_shape=(1,)),
            layers.Dense(128, activation="gelu"),
            layers.Dense(64, activation="gelu"),
            layers.Dense(1, activation="linear"),
        ], name="entropy_net")

        self.c_var = tf.Variable(central_charge, trainable=True, dtype=tf.float32, name="central_charge")

    def call(self, ell_over_a, training=False):
        log_ratio = tf.math.log(tf.cast(ell_over_a, tf.float32) + 1e-10)
        predicted_entropy = self.net(log_ratio, training=training)
        theory_entropy = (self.c_var / 3.0) * log_ratio
        return predicted_entropy, theory_entropy

    def physics_informed_loss(self, ell_over_a, S_measured):
        S_pred, S_theory = self(ell_over_a, training=True)
        data_loss = tf.reduce_mean(tf.square(S_pred - tf.cast(S_measured, tf.float32)))
        physics_loss = tf.reduce_mean(tf.square(S_pred - S_theory))
        return data_loss + 0.1 * physics_loss

    def extract_central_charge(self) -> float:
        return float(self.c_var.numpy())


class OTOCNet(keras.Model):
    def __init__(self, n_sites: int, central_charge: float):
        super().__init__()
        self.n = n_sites
        self.c = central_charge

        self.time_encoder = keras.Sequential([
            layers.Dense(32, activation="gelu"),
            layers.Dense(64, activation="gelu"),
        ])
        self.site_encoder = keras.Sequential([
            layers.Dense(32, activation="gelu"),
            layers.Dense(64, activation="gelu"),
        ])
        self.otoc_head = keras.Sequential([
            layers.Dense(64, activation="gelu"),
            layers.Dense(32, activation="gelu"),
            layers.Dense(1, activation="sigmoid"),
        ])
        self.lambda_L = tf.Variable(1.0, trainable=True, dtype=tf.float32)

    def call(self, inputs, training=False):
        t = inputs[:, :1]
        x = inputs[:, 1:]
        t_feat = self.time_encoder(t, training=training)
        x_feat = self.site_encoder(x, training=training)
        combined = tf.concat([t_feat, x_feat], axis=-1)
        F_pred = self.otoc_head(combined, training=training)
        F_theory = 1.0 - tf.exp(-tf.abs(self.lambda_L) * t)
        return F_pred, F_theory

    def chaos_bound_regularizer(self, temperature: float) -> tf.Tensor:
        kB = 1.380649e-23
        hbar = 1.054571817e-34
        bound = 2.0 * np.pi * kB * temperature / hbar
        violation = tf.maximum(0.0, tf.abs(self.lambda_L) - bound)
        return violation**2


def build_decoherence_model(
    input_dim: int,
    central_charge: float,
    hidden_dims: Optional[List[int]] = None,
    learning_rate: float = 1e-3,
) -> keras.Model:
    model = HolographicDecoherenceNet(
        central_charge=central_charge,
        hidden_dims=hidden_dims,
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss={
            "decoherence_rate": keras.losses.MeanSquaredLogarithmicError(),
            "enhancement_factor": keras.losses.MeanAbsoluteError(),
        },
        metrics={
            "decoherence_rate": [keras.metrics.MeanAbsoluteError()],
            "enhancement_factor": [keras.metrics.MeanAbsoluteError()],
        },
    )
    return model


class HolographicTrainer:
    def __init__(self, model: keras.Model, config: dict):
        self.model = model
        self.config = config
        self.history = {}

    def train(
        self,
        X_train: np.ndarray, y_train: dict,
        X_val: np.ndarray, y_val: dict,
        epochs: int = 100, batch_size: int = 32,
    ) -> dict:
        callbacks = [
            keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=8, min_lr=1e-6),
            keras.callbacks.ModelCheckpoint(
                self.config.get("checkpoint_path", "model_best.keras"),
                save_best_only=True,
            ),
        ]
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )
        self.history = history.history
        return self.history

    def evaluate(self, X_test: np.ndarray, y_test: dict) -> dict:
        results = self.model.evaluate(X_test, y_test, return_dict=True)
        return results