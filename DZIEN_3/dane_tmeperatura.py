import numpy as np
import pandas as pd

# Parametry
n_days = 3000
np.random.seed(42)

# Czas: kolejne dni
days = np.arange(n_days)

# Sezonowość + trend + szum (np. temperatura w ciągu roku + cieplejszy klimat)
temperatures = (
    10 + 10 * np.sin(2 * np.pi * days / 365.25)    # sezonowość roczna
    + 0.003 * days                                 # lekki trend globalnego ocieplenia
    + np.random.normal(0, 2, n_days)               # szum
)

df = pd.DataFrame({
    'day': days,
    'temperature': temperatures
})
df.to_csv('temperatury_miesiac.csv', index=False)
print(df.head())
