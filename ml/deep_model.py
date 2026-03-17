"""
Deep Learning Model (LSTM + Dense Neural Network)
untuk prediksi arah candle dengan TensorFlow/Keras
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (
        LSTM, Dense, Dropout, BatchNormalization,
        Input, Bidirectional, Conv1D, MaxPooling1D, Flatten, Attention
    )
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.regularizers import l2
    TF_AVAILABLE = True
    # Matikan log TF yang berisik
    tf.get_logger().setLevel("ERROR")
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
except ImportError:
    TF_AVAILABLE = False

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
from config import *

# ─── KONFIGURASI LSTM ─────────────────────────────────────
LSTM_SEQUENCE_LEN  = 30    # Panjang urutan input (30 candle sebelumnya)
LSTM_EPOCHS        = 50
LSTM_BATCH_SIZE    = 32
LSTM_UNITS_1       = 128
LSTM_UNITS_2       = 64
DENSE_UNITS        = 32
DROPOUT_RATE       = 0.3
LABEL_THRESHOLD_DL = 0.0008   # Sama dengan ML model

FEATURE_COLS_DL = [
    "rsi", "macd", "histogram", "adx", "di_pos", "di_neg",
    "stoch_k", "stoch_d", "bb_pct", "bb_bw", "atr",
    f"ema_{EMA_FAST}", f"ema_{EMA_SLOW}", f"ema_{EMA_TREND}",
    "price_change", "volatility", "candle_pat",
]


class LSTMPredictor:
    """
    Bidirectional LSTM + CNN hybrid untuk prediksi candle.
    Lebih baik dari Random Forest untuk data time series berurutan.
    """

    def __init__(self, timeframe: str = "1h"):
        if not TF_AVAILABLE:
            print("[!] TensorFlow tidak tersedia. Install: pip install tensorflow")
            self.available = False
            return
        self.available  = True
        self.timeframe  = timeframe
        self.scaler     = MinMaxScaler(feature_range=(0, 1))
        self.model      = None
        self.trained    = False
        self.accuracy   = 0.0
        self.history    = None
        self.thresh     = LABEL_THRESHOLD_DL
        self.feature_names = []

    def _get_features(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in FEATURE_COLS_DL if c in df.columns]
        self.feature_names = cols
        return df[cols].copy().fillna(0)

    def _make_sequences(self, X: np.ndarray, y: np.ndarray):
        """Buat sequence sliding window untuk LSTM"""
        Xs, ys = [], []
        for i in range(LSTM_SEQUENCE_LEN, len(X)):
            Xs.append(X[i - LSTM_SEQUENCE_LEN:i])
            ys.append(y[i])
        return np.array(Xs), np.array(ys)

    def _make_labels(self, df: pd.DataFrame) -> pd.Series:
        fwd_ret = df["Close"].shift(-1) / df["Close"] - 1
        label   = pd.Series(np.nan, index=df.index)
        label[fwd_ret >  self.thresh] = 1
        label[fwd_ret < -self.thresh] = 0
        return label

    def _build_model(self, n_features: int):
        """Bidirectional LSTM + Conv1D hybrid"""
        inputs = Input(shape=(LSTM_SEQUENCE_LEN, n_features))

        # Conv1D branch - tangkap pola lokal
        x = Conv1D(64, kernel_size=3, activation="relu", padding="same",
                   kernel_regularizer=l2(0.001))(inputs)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(DROPOUT_RATE)(x)

        # Bidirectional LSTM - tangkap tren jangka panjang
        x = Bidirectional(LSTM(LSTM_UNITS_1, return_sequences=True,
                                dropout=DROPOUT_RATE, recurrent_dropout=0.1))(x)
        x = BatchNormalization()(x)
        x = Bidirectional(LSTM(LSTM_UNITS_2, return_sequences=False,
                                dropout=DROPOUT_RATE))(x)
        x = BatchNormalization()(x)

        # Dense layers
        x = Dense(DENSE_UNITS, activation="relu", kernel_regularizer=l2(0.001))(x)
        x = Dropout(DROPOUT_RATE)(x)
        x = Dense(16, activation="relu")(x)
        output = Dense(1, activation="sigmoid")(x)

        model = Model(inputs=inputs, outputs=output)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def train(self, df: pd.DataFrame) -> dict:
        if not self.available:
            return {"error": "TensorFlow tidak tersedia"}

        # Label
        y_raw = self._make_labels(df)
        X_raw = self._get_features(df)

        # Filter sideways
        valid   = y_raw.notna()
        X_clean = X_raw[valid].values
        y_clean = y_raw[valid].values

        n_sideways = int(len(df) - valid.sum())

        # Scale fitur
        X_scaled = self.scaler.fit_transform(X_clean)

        # Buat sequences
        Xs, ys = self._make_sequences(X_scaled, y_clean)

        # Split train/test (time-series aware)
        split   = int(len(Xs) * ML_TRAIN_SPLIT)
        X_train = Xs[:split]
        X_test  = Xs[split:]
        y_train = ys[:split]
        y_test  = ys[split:]

        # Build & train
        n_features  = X_train.shape[2]
        self.model  = self._build_model(n_features)

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, verbose=0),
        ]

        print(f"     [DL] Training LSTM ({LSTM_EPOCHS} epochs max, early stop)...")
        self.history = self.model.fit(
            X_train, y_train,
            epochs=LSTM_BATCH_SIZE,
            batch_size=LSTM_BATCH_SIZE,
            validation_split=0.15,
            callbacks=callbacks,
            verbose=0,
        )

        # Evaluate
        y_proba = self.model.predict(X_test, verbose=0).flatten()
        y_pred  = (y_proba >= 0.5).astype(int)

        # Confident accuracy (proba > 0.6 atau < 0.4)
        conf_mask     = (y_proba >= 0.60) | (y_proba <= 0.40)
        conf_accuracy = accuracy_score(y_test[conf_mask], y_pred[conf_mask]) if conf_mask.sum() > 0 else 0

        self.accuracy = accuracy_score(y_test, y_pred)
        self.trained  = True

        report = classification_report(y_test, y_pred, target_names=["SELL", "BUY"], zero_division=0)

        return {
            "accuracy":          round(self.accuracy * 100, 2),
            "conf_accuracy":     round(conf_accuracy * 100, 2),
            "conf_coverage":     round(conf_mask.mean() * 100, 1),
            "n_train":           len(X_train),
            "n_test":            len(X_test),
            "n_sideways_removed":n_sideways,
            "report":            report,
            "model_params":      self.model.count_params(),
        }

    def predict(self, df: pd.DataFrame) -> dict:
        if not self.available or not self.trained:
            return {"direction": "UNKNOWN", "confidence": 0.0, "proba_buy": 0.5, "proba_sell": 0.5}

        X_raw    = self._get_features(df)
        X_scaled = self.scaler.transform(X_raw.fillna(0).values)

        # Ambil sequence terakhir
        if len(X_scaled) < LSTM_SEQUENCE_LEN:
            return {"direction": "UNKNOWN", "confidence": 0.0}

        seq    = X_scaled[-LSTM_SEQUENCE_LEN:].reshape(1, LSTM_SEQUENCE_LEN, -1)
        proba  = float(self.model.predict(seq, verbose=0)[0][0])

        proba_buy  = round(proba * 100, 1)
        proba_sell = round((1 - proba) * 100, 1)
        direction  = "BUY" if proba >= 0.5 else "SELL"
        confidence = max(proba_buy, proba_sell)
        uncertain  = confidence < 60.0

        return {
            "direction":  "WAIT" if uncertain else direction,
            "confidence": confidence,
            "proba_buy":  proba_buy,
            "proba_sell": proba_sell,
            "uncertain":  uncertain,
        }
