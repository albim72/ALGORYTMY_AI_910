import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- 1) Dane
df = pd.read_csv('temperatures_lublin_2025.csv')
# zakładam, że kolumna 'day' zaczyna się od 1 i rośnie bez dziur
day = df['day'].values.astype(float)
y = df['temperature'].values.astype(float)

# --- 2) Cechy cykliczne Fouriera (harmoniki sezonowe)
period = 365.0
harmonics = 3  # 1–3 zwykle wystarcza dla temperatur
X_cols = []
for k in range(1, harmonics+1):
    X_cols.append(np.sin(2*np.pi*k*day/period))
    X_cols.append(np.cos(2*np.pi*k*day/period))
X = np.column_stack(X_cols)

# --- 3) Skalowanie (lepszy StandardScaler niż MinMax przy sin/cos)
sx = StandardScaler().fit(X)
sy = StandardScaler().fit(y.reshape(-1,1))
Xz = sx.transform(X)
yz = sy.transform(y.reshape(-1,1))

# --- 4) Podział czasowy (bez mieszania)
split_idx = int(len(Xz)*0.9)
X_train, X_test = Xz[:split_idx], Xz[split_idx:]
y_train, y_test = yz[:split_idx], yz[split_idx:]

# --- 5) MLP z nieliniowością (mały, żeby nie przeuczyć)
tf.keras.utils.set_random_seed(42)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(32, activation='tanh'),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')

es = tf.keras.callbacks.EarlyStopping(patience=50, restore_best_weights=True)
hist = model.fit(X_train, y_train, epochs=2000, batch_size=64,
                 validation_data=(X_test, y_test), verbose=0, callbacks=[es])

# --- 6) Prognoza na przyszłość (np. 180 dni)
last_day = int(day.max())
future_days = np.arange(last_day+1, last_day+1+180, dtype=float)

# Z tych samych cech fourierowskich budujemy wejście dla przyszłości
Xf_cols = []
for k in range(1, harmonics+1):
    Xf_cols.append(np.sin(2*np.pi*k*future_days/period))
    Xf_cols.append(np.cos(2*np.pi*k*future_days/period))
Xf = np.column_stack(Xf_cols)
Xf_z = sx.transform(Xf)

pred_z = model.predict(Xf_z, verbose=0)
pred = sy.inverse_transform(pred_z)

# --- 7) Wykres: ostatnie 365 dni + prognoza
plt.figure(figsize=(11,6))
mask_last = day > last_day-365
plt.plot(day[mask_last], y[mask_last], 'bo-', lw=1, ms=3, label='Rzeczywiste (ostatni rok)')
plt.plot(future_days, pred[:,0], 'r--', lw=2, label='Prognoza (MLP + cechy Fouriera)')
plt.xlabel('Dzień'); plt.ylabel('Temperatura'); plt.title('Prognoza sezonowa (ciągłość sinusoidy)')
plt.legend(); plt.tight_layout(); plt.show()

# (Opcjonalnie) wykres straty
plt.plot(hist.history['loss'], label='train'); plt.plot(hist.history['val_loss'], label='val')
plt.yscale('log'); plt.title('MSE'); plt.legend(); plt.show()
