import numpy as np

class SimpleNeuralNetwork:
    # def __new__(cls):
    #     return super().__new__(cls)

    def __init__(self):
        np.random.seed(1)
        self.weights = 2*np.random.rand(3, 1)-1

    def __repr__(self):
        return f"SimpleNeuralNetwork(weights=\n{self.weights})"
