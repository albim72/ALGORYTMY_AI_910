import matplotlib.pyplot as plt


# Wyświetlenie pierwszych 5 obrazów testowych
plt.figure(figsize=(10, 2))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(x_test[i], cmap="gray")
    plt.title(f"Etykieta: {y_test[i]}")
    plt.axis("off")

plt.show()
