import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

series = y.reshape(-1,1)
sy = StandardScaler().fit(series)
series_z = sy.transform(series).astype(np.float32)

# okno 30 dni prognozuje 1 dzień do przodu
win = 30
def make_windows(a, w):
    X, Y = [], []
    for i in range(len(a)-w):
        X.append(a[i:i+w])
        Y.append(a[i+w])
    return np.array(X), np.array(Y)

Xw, Yw = make_windows(series_z, win)        # shape: (N, 30, 1)
split_idx = int(0.9*len(Xw))
Xtr, Xte = Xw[:split_idx], Xw[split_idx:]
Ytr, Yte = Yw[:split_idx], Yw[split_idx:]

tf.keras.utils.set_random_seed(42)
model_seq = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(win,1)),
    tf.keras.layers.Conv1D(32, 5, activation='relu', padding='causal'),
    tf.keras.layers.LSTM(64, return_sequences=False),
    tf.keras.layers.Dense(1)
])
model_seq.compile(optimizer='adam', loss='mse')
model_seq.fit(Xtr, Ytr, epochs=200, batch_size=64, validation_data=(Xte, Yte), verbose=0)

# iteracyjna prognoza na 180 dni (autoregresja)
window = series_z[-win:].copy()
future = []
for _ in range(180):
    yhat = model_seq.predict(window.reshape(1,win,1), verbose=0)[0,0]
    future.append(yhat)
    window = np.vstack([window[1:], [yhat]])

future = sy.inverse_transform(np.array(future).reshape(-1,1))
future_days = np.arange(int(day.max())+1, int(day.max())+1+len(future))

plt.figure(figsize=(11,6))
plt.plot(day[-365:], y[-365:], 'bo-', ms=3, label='Rzeczywiste (ostatni rok)')
plt.plot(future_days, future[:,0], 'g--', lw=2, label='Prognoza (Conv1D+LSTM, okna)')
plt.legend(); plt.title('Prognoza sekwencyjna (zachowanie sinusoidy)'); plt.show()
