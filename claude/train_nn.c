// train_nn.c - Main Neural Network Training Program
// Compile: gcc train_nn.c nn_lib.c -o train_nn -lm -O3 -march=native -ffast-math
// Run: ./train_nn

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "nn_lib.h"

// Network architecture
#define HID1 128
#define HID2 64

// Training hyperparameters
#define EPOCHS 30
#define BATCH_SIZE 64
#define LEARNING_RATE 0.001f
#define MOMENTUM 0.9f
#define L2_LAMBDA 0.0001f

int main() {
    srand(time(NULL));
    
    // Load training data
    int num_train_images, num_train_labels;
    float *train_images = read_mnist_images("../files/train-images-idx3-ubyte", &num_train_images);
    unsigned char *train_labels = read_mnist_labels("../files/train-labels-idx1-ubyte", &num_train_labels);
    
    if (!train_images || !train_labels || num_train_images != num_train_labels) {
        fprintf(stderr, "Error loading training data\n");
        fprintf(stderr, "Make sure MNIST data files are in the current directory\n");
        free(train_images);
        free(train_labels);
        return 1;
    }
    
    printf("Loaded %d training samples\n", num_train_images);
    
    // Load test data
    int num_test_images, num_test_labels;
    float *test_images = read_mnist_images("../files/t10k-images-idx3-ubyte", &num_test_images);
    unsigned char *test_labels = read_mnist_labels("../files/t10k-labels-idx1-ubyte", &num_test_labels);
    
    if (!test_images || !test_labels || num_test_images != num_test_labels) {
        fprintf(stderr, "Error loading test data\n");
        free(train_images);
        free(train_labels);
        free(test_images);
        free(test_labels);
        return 1;
    }
    
    printf("Loaded %d test samples\n\n", num_test_images);
    
    // Create network
    int layer_sizes[] = {INPUT_SIZE, HID1, HID2, OUTPUT_SIZE};
    Network *net = create_network(layer_sizes, 4);
    
    if (!net) {
        fprintf(stderr, "Error creating network\n");
        free(train_images);
        free(train_labels);
        free(test_images);
        free(test_labels);
        return 1;
    }
    
    // Set hyperparameters
    net->learning_rate = LEARNING_RATE;
    net->momentum = MOMENTUM;
    net->l2_lambda = L2_LAMBDA;
    
    // Train network
    train_network(net, train_images, train_labels, num_train_images, EPOCHS, BATCH_SIZE);
    
    // Validate on test set
    printf("\nValidating on test set...\n");
    float test_accuracy = validate_network(net, test_images, test_labels, num_test_images);
    printf("Test accuracy: %.2f%%\n", test_accuracy);
    
    // Save weights
    if (save_network(net, "weights.bin")) {
        printf("\nWeights saved to weights.bin\n");
    } else {
        fprintf(stderr, "Error saving weights\n");
    }
    
    // Clean up
    free_network(net);
    free(train_images);
    free(train_labels);
    free(test_images);
    free(test_labels);
    
    return 0;
}
