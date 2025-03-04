
# Micrograd in Java

This repository is a Java implementation of Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd). Micrograd is a simple autograd engine which supports automatic differentiation and includes basic arithmetic operations, activation functions, and a multi-layer perceptron (MLP) for binary classification tasks.

## Features

- **Automatic Differentiation:** Reverse-mode autodiff for computing gradients.
- **Arithmetic Operations:** Supports addition, subtraction, multiplication, division, and exponentiation.
- **Activation Functions:** Implements tanh, ReLU, and exponential functions.
- **Neural Network Modules:** Includes basic building blocks:
  - **Neuron:** Single neuron with optional nonlinearity.
  - **Layer:** A collection of neurons.
  - **MLP:** A multi-layer perceptron constructed from multiple layers.
- **Training Example:** `TestAutograd.java` demonstrates a full training loop including:
  - Synthetic data generation for binary classification.
  - Forward pass and loss computation (mean squared error with tanh activation).
  - Backward pass to compute gradients.
  - Gradient descent updates.
  - Final prediction output.

## File Structure

- **Value.java:**  
  Implements the `Value` class representing a node in the computational graph. Contains operations, activation functions, and the backward pass.

- **Module.java:**  
  Abstract base class for neural network modules. Provides methods for parameter management and zeroing gradients.

- **Neuron.java:**  
  Implements a single neuron that computes a weighted sum of its inputs with an optional nonlinearity.

- **Layer.java:**  
  Represents a layer composed of multiple neurons.

- **MLP.java:**  
  Implements a multi-layer perceptron by stacking layers together.

- **TestAutograd.java:**  
  Contains an example of using the autograd engine:
  - Generates synthetic data with two features and target labels (1.0 for positive, -1.0 for negative).
  - Constructs an MLP with one hidden layer.
  - Performs a forward pass with tanh as the output activation.
  - Computes the mean squared error (MSE) loss.
  - Runs a backward pass to compute gradients.
  - Updates parameters via gradient descent.
  - Prints training loss and final binary predictions.

## Requirements

- Java Development Kit (JDK 8 or higher)

## How to Compile and Run

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/micrograd-java.git
   cd micrograd-java
   ```

2. **Compile the Source Files:**

   ```bash
   javac Value.java Module.java Neuron.java Layer.java MLP.java TestAutograd.java
   ```

3. **Run the Test Example:**

   ```bash
   java TestAutograd
   ```

   You should see output showing the training loss over epochs and the final predictions, for example:

   ```
   Epoch 0 Loss: 0.793036169161181
   Epoch 10 Loss: 0.7147676887206238
   ...
   Final predictions on training data:
   Input: (0.7275636800328681, 0.6832234717598454)  Prediction: 1 (tanh output: 0.8586039831774307)
   ...
   ```

## Customization

- **Network Architecture:**  
  Modify `MLP.java` or `TestAutograd.java` to experiment with different numbers of layers or neurons.

- **Training Parameters:**  
  Adjust the learning rate, number of epochs, or even data generation logic in `TestAutograd.java` as needed.

## Acknowledgements

Inspired by [micrograd](https://github.com/karpathy/micrograd) by Andrej Karpathy. This implementation aims to replicate its functionality using Java.
