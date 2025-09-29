import numpy as np
from simplenn import SimpleNeuralNetwork

network  = SimpleNeuralNetwork()
print(network)

#dane wejściowe
train_inputs = np.array([[1,1,0],[1,1,1],[1,1,0],[1,0,0],[0,1,1],[0,1,0],[0,0,0]])
train_outputs = np.array([[1,0,1,1,0,1,0]]).T
train_iters = 50_000

#trening sieci neuronowej
network.train(train_inputs,train_outputs,train_iters)
print(f"\nwagi po treningu:\n{network.weights}")

#testowanie
print(f"\nprzykładowe wejście: {train_inputs[0]}")
print(f"\nprzykładowe wyjście: {train_outputs[0]}")
print(f"\nwyjście po wykonaniu: {network.propagation(train_inputs[0])} \n\n")

#predykcja
test_data = np.array([[1,1,1],[1,0,0],[0,1,1],[0,1,0],[0,0,1],[0,0,0]])
print(f"\npredykcja z wykorzystamniem sieci neuronowej")
for data in test_data:
    print(f"przykładowe dane: {data}")
    print(f"wyjście predykcji: {network.propagation(data)}\n\n")

