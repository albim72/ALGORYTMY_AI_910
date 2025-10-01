import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Wczytaj dane
df = pd.read_csv('temperatury_miesiac.csv')

X = df[['day']].values
y = df['temperature'].values

# Skalowanie cechy (ważne dla sieci!)
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# Podział na train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.1, shuffle=False)

# Model: pojedyncza warstwa – klasyczna regresja liniowa
model_linear = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,))
])
model_linear.compile(optimizer='adam', loss='mse')
history_linear = model_linear.fit(X_train, y_train, epochs=100, verbose=0, validation_data=(X_test, y_test))

# Wizualizacja przebiegu uczenia
plt.plot(history_linear.history['loss'], label='train_loss')
plt.plot(history_linear.history['val_loss'], label='val_loss')
plt.title("Regresja liniowa: strata MSE")
plt.xlabel("Epoka")
plt.ylabel("MSE")
plt.legend()
plt.show()



# Głęboka sieć: 3 warstwy + nieliniowość
model_deep = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
])
model_deep.compile(optimizer='adam', loss='mse')
history_deep = model_deep.fit(X_train, y_train, epochs=100, verbose=0, validation_data=(X_test, y_test))

plt.plot(history_deep.history['loss'], label='train_loss')
plt.plot(history_deep.history['val_loss'], label='val_loss')
plt.title("Głęboka sieć: strata MSE")
plt.xlabel("Epoka")
plt.ylabel("MSE")
plt.legend()
plt.show()


# Prognoza na 30 dni po ostatnim dniu
future_days = np.arange(n_days, n_days + 30).reshape(-1, 1)
future_days_scaled = scaler_X.transform(future_days)

# Prognozy
pred_linear = model_linear.predict(future_days_scaled)
pred_deep = model_deep.predict(future_days_scaled)

# Odwracamy skalowanie
pred_linear_real = scaler_y.inverse_transform(pred_linear)
pred_deep_real = scaler_y.inverse_transform(pred_deep)

# Wizualizacja – ostatnie 60 dni + predykcja na 30 dni
plt.figure(figsize=(10,6))
plt.plot(df['day'].iloc[-60:], df['temperature'].iloc[-60:], 'bo-', label='Rzeczywiste')
plt.plot(np.arange(n_days, n_days + 30), pred_linear_real, 'r--', label='Regresja liniowa')
plt.plot(np.arange(n_days, n_days + 30), pred_deep_real, 'g--', label='Sieć głęboka')
plt.xlabel('Dzień')
plt.ylabel('Temperatura')
plt.title('Prognoza temperatury na 30 dni (sieć liniowa i głęboka)')
plt.legend()
plt.show()
