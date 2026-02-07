#  Neural Network from Scratch 

This project implements a simple feedforward neural network from scratch using Python and NumPy.
No in-built deep learning libraries such as TensorFlow, Keras, PyTorch, or Scikit-learn are used.

The main goal of this project is to demonstrate the core concepts of neural networks, including:
- Weight Initialization
- Forward Propagation
- Error (Loss) Calculation
- Backpropagation
- Gradient Descent based weight updates

##  Neural Network Architecture
- Input Layer: 2 neurons  
- Hidden Layer:3 neurons (Sigmoid activation)  
- Output Layer:1 neuron (Sigmoid activation)

## Dataset Used
A simple NAND logic gate dataset is used for training:

This small dataset is sufficient to demonstrate learning behavior in a neural network.

## Key Concepts Implemented

### 1. Weight Initialization
Weights and biases are initialized with small fixed values to start the learning process.

### 2. Forward Propagation
Inputs are passed through the network using weighted sums and the sigmoid activation function to produce output.

### 3. Error Calculation
The error is calculated as the difference between the actual output and the predicted output.
Mean Squared Error (MSE) is used as the loss function.

### 4. Backpropagation
The error is propagated backward through the network to compute gradients using the chain rule.

### 5. Gradient Descent
Weights and biases are updated using gradient descent to minimize the loss.

## Files in Repository
- `neural_network_from_scratch.ipynb`  
  → Google Colab / Jupyter Notebook containing the full implementation of the neural network. 

##  How to Run the Code
1. Open the notebook in **Google Colab** or **Jupyter Notebook**
2. Run all cells sequentially
3. Observe:
   - Loss printed during training
   - Final predictions after training

##  Output
After training for 10,000 epochs, the neural network successfully learns the NAND logic and produces correct predictions.

## Author
Soha Sameer Sayyed 


