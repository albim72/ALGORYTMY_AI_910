# forecast_lstm.py
# Prognoza temperatury: sekwencyjny model Conv1D+LSTM
# - okna czasowe (autoregresja)
# - sezonowość przez kanały sin/cos dnia roku
# - stabilne skalowanie (StandardScaler)
# - podział czasowy bez mieszania

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# =========================
# 1) Wczytanie i przygotowanie danych
# =========================
CSV_PATH = "temperatures_lublin_2025.csv"  # wymagane kolumny: day, temperature
assert Path(CSV_PATH).exists(), f"Brak pliku: {CSV_PATH}"

df = pd.read_csv(CSV_PATH)

# sanity checks
for col in ["day", "temperature"]:
    assert col in df.columns, f"Brak kolumny '{col}' w {CSV_PATH}"

# sort i reset indeksów
df = df.sort_values("day").reset_index(drop=True)

# Jeżeli są braki dni — uzupełnij przez reindeksację i interpolację
day_min, day_max = int(df["day"].min()), int(df["day"].max())
full_idx = pd.RangeIndex(day_min, day_max + 1, name="day")
df = df.set_index("day").reindex(full_idx)

# Interpolacja temperatur (jeśli są NaN po reindeksacji)
df["temperature"] = df["temperature"].interpolate(method="linear").ffill().bfill()

# Dzień roku (1..365) cyklicznie
d = df.index.to_numpy(dtype=np.int64)
doy = ((d - 1) % 365) + 1  # 1..365
sin_doy = np.sin(2 * np.pi * doy / 365.0)
cos_doy = np.cos(2 * np.pi * doy / 365.0)

temp = df["temperature"].to_numpy(dtype=np.float32)

# =========================
# 2) Skalowanie
# =========================
# Skalujemy tylko temperaturę (kanały sin/cos już są w [-1,1])
sy = StandardScaler()
temp_z = sy.fit_transform(temp.reshape(-1, 1)).astype(np.float32).ravel()

# Złożenie macierzy cech (3 kanały: temp_z, sin, cos)
# Uwaga: temp_z to seria; sin/cos to deterministyczne cechy wejściowe na każdy dzień
X_all = np.stack([temp_z, sin_doy.astype(np.float32), cos_doy.astype(np.float32)], axis=-1)
# shape: (T, 3)

# =========================
# 3) Budowa okien czasowych
# =========================
def make_windows(X, y, win):
    """
    X: (T, C)   y: (T,)
    zwraca:
      Xw: (N, win, C)
      Yw: (N,)
    """
    T = len(y)
    Xw = []
    Yw = []
    for i in range(T - win):
        Xw.append(X[i:i+win, :])
        Yw.append(y[i+win])
    return np.array(Xw, dtype=np.float32), np.array(Yw, dtype=np.float32)

WIN = 30  # okno 30 dni przewiduje 1 dzień do przodu
Xw, Yw = make_windows(X_all, temp_z, WIN)  # Xw: (N, 30, 3), Yw: (N,)

# =========================
# 4) Podział czasowy train/val
# =========================
split = int(0.9 * len(Xw))  # ostatnie 10% jako walidacja
Xtr, Xva = Xw[:split], Xw[split:]
Ytr, Yva = Yw[:split], Yw[split:]

# =========================
# 5) Model Conv1D + LSTM
# =========================
tf.keras.utils.set_random_seed(42)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(WIN, Xtr.shape[-1])),  # (30, 3)
    tf.keras.layers.Conv1D(32, kernel_size=5, activation="relu", padding="causal"),
    tf.keras.layers.LSTM(64, return_sequences=False),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")

callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=20, verbose=0),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=60, restore_best_weights=True, verbose=0),
]

hist = model.fit(
    Xtr, Ytr,
    validation_data=(Xva, Yva),
    epochs=1000,
    batch_size=64,
    verbose=0,
    callbacks=callbacks
)

# =========================
# 6) Autoregresyjna prognoza N dni
# =========================
FORECAST_DAYS = 180  # horyzont prognozy
last_day = d[-1]
future_days = np.arange(last_day + 1, last_day + 1 + FORECAST_DAYS, dtype=np.int64)

# przygotuj okno startowe (ostatnie 30 dni)
window = X_all[-WIN:, :].copy()  # (30, 3)
pred_z = []

for t_i, day_i in enumerate(future_days):
    # przyszłe cechy sin/cos znane z kalendarza:
    doy_i = ((day_i - 1) % 365) + 1
    sin_i = np.sin(2 * np.pi * doy_i / 365.0).astype(np.float32)
    cos_i = np.cos(2 * np.pi * doy_i / 365.0).astype(np.float32)

    # przewidź temp_z na jutro na podstawie bieżącego okna
    yhat_z = model.predict(window.reshape(1, WIN, 3), verbose=0)[0, 0]
    pred_z.append(yhat_z)

    # zaktualizuj okno: przesuwamy o 1 i dokładamy nowy wiersz [temp_z_pred, sin, cos]
    new_row = np.array([yhat_z, sin_i, cos_i], dtype=np.float32)
    window = np.vstack([window[1:], new_row])

pred_z = np.array(pred_z, dtype=np.float32).reshape(-1, 1)
pred = sy.inverse_transform(pred_z).ravel()  # powrót do temperatur rzeczywistych

# =========================
# 7) Wizualizacja
# =========================
plt.figure(figsize=(11, 6))
# ostatni rok danych dla kontekstu
mask_last_year = d >= (last_day - 364)
plt.plot(d[mask_last_year], temp[mask_last_year], 'bo-', ms=3, lw=1, label='Rzeczywiste (ostatnie 365 dni)')
plt.plot(future_days, pred, 'r--', lw=2, label=f'Prognoza {FORECAST_DAYS} dni (Conv1D+LSTM)')
plt.xlabel("Dzień")
plt.ylabel("Temperatura")
plt.title("Prognoza temperatury – model sekwencyjny (z sezonowością sin/cos)")
plt.legend()
plt.tight_layout()
plt.show()

# Krzywe uczenia (log-MSE)
plt.figure(figsize=(9, 4.5))
plt.plot(hist.history['loss'], label='train')
plt.plot(hist.history['val_loss'], label='val')
plt.yscale('log')
plt.title('Strata MSE (log)')
plt.xlabel('Epoka')
plt.legend()
plt.tight_layout()
plt.show()

print("Gotowe. Prognoza dni:", len(future_days), " Od dnia:", future_days[0], " do:", future_days[-1])
