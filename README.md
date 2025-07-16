# Neural Network from Scratch

This project implements a basic neural network from scratch using Python and NumPy, inspired by the Neural Networks from Scratch playlist by Vizura Labs. The code demonstrates the construction of a single neuron, a fully connected (dense) layer, multiple stacked layers, activation functions (ReLU and SoftMax), and loss calculation using categorical cross-entropy.

## Project Overview

The code is structured in a Jupyter Notebook (`Neural_net_scratch.ipynb`) and covers the following components:

1. **Single Neuron Implementation**:
   - Models a single neuron with inputs (`x`), weights (`w`), bias (`b`), and a sigmoid activation function.
   - Example output: `0.9002495108803148`.

2. **Fully Connected Layer**:
   - Implements a dense layer with 5 neurons and 3 input features.
   - Example weights and output for a fully connected layer are provided.

3. **Multi-Layer Neural Network**:
   - Stacks multiple dense layers with matrix multiplications.
   - Example with input `x`, weights `w1` and `w2`, biases `b1` and `b2`, and ReLU activation:
     ```python
     z1 = np.dot(x, w1.T) + b1
     z2 = np.dot(z1, w2.T) + b2
     a = np.maximum(0, z2)
     ```
   - Sample output:
     ```
     [[19.25  7.24 14.47]
      [17.9   5.93 13.64]
      [16.71  6.33 12.63]]
     ```

4. **Non-Linear Training Data**:
   - Generates spiral data using the `nnfs` library for training neural networks.
   - Visualizes the spiral dataset using Matplotlib.

5. **Dense Layer Class**:
   - Implements a `Dense` class for fully connected layers with random weight initialization and forward pass:
     ```python
     class Dense:
         def __init__(self, n_inputs, n_neurons):
             self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
             self.biases = np.zeros((1, n_neurons))
         def forward(self, inputs):
             self.output = np.dot(inputs, self.weights) + self.biases
     ```

6. **Activation Functions**:
   - **ReLU**: Applies `np.maximum(0, inputs)` for non-linearity.
   - **SoftMax**: Normalizes outputs into probabilities using exponential normalization.
   - Example outputs for ReLU and SoftMax are provided for spiral data.

7. **Loss Calculation**:
   - Implements categorical cross-entropy loss for both sparse and one-hot encoded labels.
   - Example loss calculation:
     ```python
     loss_fn = Loss_CategoricalCrossentropy()
     print("Loss (sparse):", loss_fn.forward(y_pred, y_true_sparse))
     print("Loss (one-hot):", loss_fn.forward(y_pred, y_true_onehot))
     ```
   - Sample output: `Loss: 0.38506088005216804`.

8. **Final Model**:
   - Combines dense layers, ReLU, and SoftMax for a simple neural network.
   - Achieves a loss of `0.0053851516` and accuracy of `0.36` on the spiral dataset.

## Dependencies

- Python 3.x
- NumPy
- Matplotlib
- nnfs (Neural Networks from Scratch library)
  ```bash
  pip install nnfs

How to Run

Clone the repository:git clone <repository-url>


Install dependencies:pip install numpy matplotlib nnfs


Open and run the Neural_net_scratch.ipynb notebook in Jupyter or Google Colab.
Follow the notebook cells to execute the code and visualize the results.

Credits
This project is heavily inspired by the Neural Networks from Scratch playlist by Vizura Labs. Their detailed explanations and step-by-step tutorials were instrumental in building this implementation. Check out their content for an in-depth understanding of neural networks:

Vizura Labs Neural Networks from Scratch Playlist
