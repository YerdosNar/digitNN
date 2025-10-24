# MNIST Digit recognition Neural Network \[[Perceptron](https://www.w3schools.com/ai/ai_perceptrons.asp)\]
---
- ## Original:
    It is the neural network that I first developed, my first attempt. 784-neuron INPUT layer, arbitrary number of neurons in 2 HIDDEN layers, and 10-neuron in the OUTPUT layer.

- ## 1d:
    It is my second attempt to optimize the network by transforming 2d matrices into 1d arrays. Spoiler: it didn't increase the training speed much (by less than 3%).

- ## Enhanced:
    It is the neural network that I worked on with ClaudeAI. It has **momentum**, **L2 regulation** as improvements.

---
## RUN (train):
Clone the repository:
```bash
git clone https://github.com/YerdosNar/digitNN.git
```
You can find run script in every directory:
```bash
./run
```
You will be prompted to input the size of each layer. And training starts.

---
## Test:
Just run `./test_nn` for testing with test data or `./draw_test` for testing by drawing your own digits.

---
## Files:
Train data, test data, font files are inside **files** directory.
