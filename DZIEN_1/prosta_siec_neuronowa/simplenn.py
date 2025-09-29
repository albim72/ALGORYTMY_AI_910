import numpy as np

class SimpleNeuralNetwork:
    # def __new__(cls):
    #     return super().__new__(cls)

    def __init__(self):
        np.random.seed(1)
        self.weights = 2*np.random.rand(3, 1)-1

    def __repr__(self):
        return f"SimpleNeuralNetwork(weights=\n{self.weights})"

    #funkcja aktywacji
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    #różniczka funkcji aktywacji
    def d_sigmoid(self, x):
        return x*(1-x)
    
    #funkcja propagacji
    def propagation(self,inputs):
        self.output = self.sigmoid(np.dot(inputs, self.weights))
        return self.output
    
    #funkcja propagacji wstecznej
    def backward_propagation(self,propagation_result,train_input,train_output):
        error = train_output - propagation_result
        d_error = error * self.d_sigmoid(propagation_result)
        d_weights = np.dot(train_input.T, d_error)
        self.weights += d_weights  
        
    #funkcja trening
    def train(self,train_input,train_output,train_iters):
        for _ in range(train_iters):
            propagation_result = self.propagation(train_input)
            self.backward_propagation(propagation_result,train_input,train_output)
        
    



