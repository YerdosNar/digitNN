// Main program for training the neural network
// This uses the functions from nn_lib.c

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "nn.h"

#define INPUT 784
#define HID1 128
#define HID2 64
#define OUTPUT 10
#define EPOCHS 30
#define BATCH_SIZE 64

int main() {
    srand(time(NULL));
    
    // Load training data
    int num_train_images, num_train_labels;
    float *train_images = read_mnist_images("train-images-idx3-ubyte", &num_train_images);
    unsigned char *train_labels = read_mnist_labels("train-labels-idx1-ubyte", &num_train_labels);
    
    if (!train_images || !train_labels || num_train_images != num_train_labels) {
        fprintf(stderr, "Error loading training data\n");
        free(train_images);
        free(train_labels);
        return 1;
    }
    
    printf("Loaded %d training samples\n", num_train_images);
    
    // Load test data
    int num_test_images, num_test_labels;
    float *test_images = read_mnist_images("t10k-images-idx3-ubyte", &num_test_images);
    unsigned char *test_labels = read_mnist_labels("t10k-labels-idx1-ubyte", &num_test_labels);
    
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
    int layer_sizes[] = {INPUT, HID1, HID2, OUTPUT};
    Network *net = create_network(layer_sizes, 4);
    
    if (!net) {
        fprintf(stderr, "Error creating network\n");
        free(train_images);
        free(train_labels);
        free(test_images);
        free(test_labels);
        return 1;
    }
    
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
