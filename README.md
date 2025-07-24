# CNN from Scratch on MNIST (NumPy) + PyTorch Comparison 🧠

This project explores the implementation of a **Convolutional Neural Network (CNN)** from the ground up using only **NumPy**. Its performance is then benchmarked against an equivalent model built with **PyTorch** on the classic MNIST dataset of handwritten digits.

The primary goal is to provide a deep, hands-on understanding of the internal mechanics of a CNN, including forward and backward propagation, while also highlighting the practical advantages of using a modern deep-learning framework like PyTorch.

---

## 🏗️ CNN Architecture Concepts

Both the NumPy and PyTorch models in this project are built using the same fundamental CNN architecture. Here are the core components:

* **Convolutional Layers (`Conv2D`)**: These are the primary building blocks for feature extraction. They apply a set of learnable filters (kernels) across the input image to create feature maps that detect patterns like edges, corners, and textures.
    * **NumPy**: Implemented from scratch, manually handling padding, stride, and the complex backpropagation calculations.
    * **PyTorch**: Uses the highly optimized `nn.Conv2d` layer, which handles all underlying operations automatically.

* **Activation Function (`ReLU`)**: The Rectified Linear Unit introduces non-linearity, allowing the network to learn complex relationships in the data. It simply returns the input if it's positive and zero otherwise (`f(x) = max(0, x)`).
    * **NumPy**: Implemented as a simple `max(0, input)` function.
    * **PyTorch**: Uses the efficient `F.relu` function.

* **Pooling Layers (`MaxPool2D`)**: These layers downsample the feature maps, reducing their spatial dimensions. This decreases the number of parameters and computational load, making the model faster and more robust to variations in the position of features.
    * **NumPy**: Manually implemented, requiring tracking the indices of the max values for use during backpropagation.
    * **PyTorch**: Uses the `nn.MaxPool2d` layer, which manages the backward pass automatically.

* **Flatten Layer**: This layer converts the 2D feature maps from the convolutional/pooling layers into a 1D vector, preparing the data to be fed into the fully connected layers.

* **Fully Connected Layers (`Dense`/`Linear`)**: These are standard neural network layers where every neuron is connected to every neuron in the previous layer. They perform classification based on the features extracted by the convolutional layers.
    * **NumPy**: The entire forward pass, backward pass, and weight/bias update logic is contained within the layer's implementation.
    * **PyTorch**: Uses the `nn.Linear` layer. Optimization (weight updates) is handled separately by a PyTorch optimizer like Adam or SGD.

* **Loss Function (`Cross-Entropy Loss`)**: This function measures how well the model is performing. It quantifies the difference between the predicted probabilities and the actual labels. The goal of training is to minimize this loss.
    * **NumPy**: Implemented within a `SoftmaxWithCrossEntropy` class, which combines the Softmax activation and the loss calculation for better numerical stability.
    * **PyTorch**: Uses the standard `nn.CrossEntropyLoss`, which is highly optimized and also combines Softmax and cross-entropy loss.

---

## 📊 NumPy vs. PyTorch: A Comparison

The performance difference between the "from scratch" implementation and the PyTorch framework is significant. The NumPy model was intentionally trained on a small subset of data due to its computational intensity.

| Metric                | NumPy (from Scratch)          | PyTorch                     |
| :-------------------- | :---------------------------- | :-------------------------- |
| **Training Data Size** | 1,000 samples                 | 60,000 samples              |
| **Training Time** | ~150 seconds / epoch          | ~120 seconds / epoch (on full data) |
| **Epochs Trained** | 10                            | 5                           |
| **Final Training Loss** | ~2.05                         | ~0.016                      |
| **Test Accuracy** | **26.80%** (on 1k samples)    | **98.92%** (on 10k samples) |
| **Final Test Loss** | ~2.115                        | N/A (not explicitly stated) |

---

## ✅ Conclusion

This project successfully demonstrates two key points:

1.  **Educational Value**: Building a CNN from scratch with NumPy offers invaluable insight into the core mechanics of deep learning, especially the backpropagation algorithm across different layer types. You get to see exactly how filters, pooling, and gradients work.

2.  **Framework Efficiency**: For any practical application, high-level frameworks like PyTorch are the standard. They are vastly more efficient, less error-prone, and allow for rapid development and scaling. The superior results of the PyTorch model are due to its optimized operations and its ability to train on the entire dataset in a reasonable amount of time.

In short, **build from scratch to learn, but use a framework to build for production**. 🚀
