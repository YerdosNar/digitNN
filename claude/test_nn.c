// test_nn.c - Test Program for Evaluating the Trained Neural Network
// Compile with: gcc test_nn.c -o test_nn -lm -O3
// Run after training: ./test_nn

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>

#include "nn.h"

#define INPUT 784
#define OUTPUT 10

// Define the functions that are needed from nn.c
// These are simple implementations just for testing

// Reverse endianness for MNIST files
int reverse_int(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

// Initialize layer (simplified version for loading only)
bool init_layer(Layer *l, int pre_size, int size) {
    l->size = size;
    l->pre_size = pre_size;
    
    l->neurons = calloc(size, sizeof(float));
    l->pre_activation = calloc(size, sizeof(float));
    l->biases = calloc(size, sizeof(float));
    l->deltas = calloc(size, sizeof(float));
    
    if (!l->neurons || !l->pre_activation || !l->biases || !l->deltas) {
        return false;
    }
    
    l->weights = malloc(size * sizeof(float*));
    if (!l->weights) {
        return false;
    }
    
    for (int i = 0; i < size; i++) {
        l->weights[i] = calloc(pre_size, sizeof(float));
        if (!l->weights[i]) {
            return false;
        }
    }
    
    return true;
}

// Free network
void free_network(Network *net) {
    if (!net) return;
    
    for (int i = 0; i < net->num_layers; i++) {
        free_layer(&net->layers[i]);
    }
    free(net->layers);
    free(net);
}

// ReLU activation
static inline float relu(float x) {
    return fmaxf(0.0f, x);
}

// Feedforward
void feedforward(const float *input, int input_size, Layer *l, bool use_relu) {
    float *pre_act = l->pre_activation;
    float *neurons = l->neurons;
    float *biases = l->biases;
    float **weights = l->weights;
    
    // Initialize with biases
    memcpy(pre_act, biases, l->size * sizeof(float));
    
    // Matrix multiplication
    for (int i = 0; i < l->size; i++) {
        float sum = pre_act[i];
        float *weight_row = weights[i];
        
        for (int j = 0; j < input_size; j++) {
            sum += weight_row[j] * input[j];
        }
        
        pre_act[i] = sum;
        neurons[i] = use_relu ? relu(sum) : sum;
    }
}

// Softmax
void softmax_stable(float *output, int size) {
    float max_val = output[0];
    for (int i = 1; i < size; i++) {
        if (output[i] > max_val) max_val = output[i];
    }
    
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        output[i] = expf(output[i] - max_val);
        sum += output[i];
    }
    
    if (sum > 0) {
        float inv_sum = 1.0f / sum;
        for (int i = 0; i < size; i++) {
            output[i] *= inv_sum;
        }
    }
}

// Forward pass
void forward_pass(Network *net, const float *input) {
    // First hidden layer
    feedforward(input, INPUT, &net->layers[0], true);
    
    // Subsequent layers
    for (int i = 1; i < net->num_layers - 1; i++) {
        feedforward(net->layers[i-1].neurons, net->layers[i-1].size, 
                   &net->layers[i], true);
    }
    
    // Output layer (no ReLU)
    int last = net->num_layers - 1;
    feedforward(net->layers[last-1].neurons, net->layers[last-1].size,
                &net->layers[last], false);
    
    // Apply softmax
    softmax_stable(net->layers[last].neurons, net->layers[last].size);
}

// Load network from file
Network* load_network(const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error: Cannot open %s\n", filename);
        return NULL;
    }
    
    int num_layers;
    if (fread(&num_layers, sizeof(int), 1, file) != 1) {
        fclose(file);
        return NULL;
    }
    
    int *sizes = malloc((num_layers + 1) * sizeof(int));
    if (fread(sizes, sizeof(int), num_layers + 1, file) != (size_t)(num_layers + 1)) {
        free(sizes);
        fclose(file);
        return NULL;
    }
    
    Network *net = malloc(sizeof(Network));
    net->num_layers = num_layers;
    net->layers = malloc(num_layers * sizeof(Layer));
    net->learning_rate = 0.001f;
    net->momentum = 0.9f;
    net->l2_lambda = 0.0001f;
    
    // Initialize layers
    for (int i = 0; i < num_layers; i++) {
        if (!init_layer(&net->layers[i], sizes[i], sizes[i + 1])) {
            fprintf(stderr, "Failed to initialize layer %d\n", i);
            free(sizes);
            fclose(file);
            free(net);
            return NULL;
        }
    }
    
    // Load weights and biases
    for (int i = 0; i < num_layers; i++) {
        Layer *l = &net->layers[i];
        if (fread(l->biases, sizeof(float), l->size, file) != (size_t)l->size) {
            fprintf(stderr, "Failed to read biases for layer %d\n", i);
            free_network(net);
            free(sizes);
            fclose(file);
            return NULL;
        }
        
        for (int j = 0; j < l->size; j++) {
            if (fread(l->weights[j], sizeof(float), l->pre_size, file) != (size_t)l->pre_size) {
                fprintf(stderr, "Failed to read weights for layer %d\n", i);
                free_network(net);
                free(sizes);
                fclose(file);
                return NULL;
            }
        }
    }
    
    free(sizes);
    fclose(file);
    return net;
}

// MNIST reading functions
float* read_mnist_images(const char *filename, int *count) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error: Cannot open %s\n", filename);
        return NULL;
    }
    
    int magic_number;
    if (fread(&magic_number, sizeof(int), 1, file) != 1) {
        fprintf(stderr, "Error: Failed to read magic number\n");
        fclose(file);
        return NULL;
    }
    magic_number = reverse_int(magic_number);
    
    if (magic_number != 2051) {
        fprintf(stderr, "Error: Invalid magic number in %s\n", filename);
        fclose(file);
        return NULL;
    }
    
    if (fread(count, sizeof(int), 1, file) != 1) {
        fprintf(stderr, "Error: Failed to read count\n");
        fclose(file);
        return NULL;
    }
    *count = reverse_int(*count);
    
    int rows, cols;
    if (fread(&rows, sizeof(int), 1, file) != 1 || 
        fread(&cols, sizeof(int), 1, file) != 1) {
        fprintf(stderr, "Error: Failed to read dimensions\n");
        fclose(file);
        return NULL;
    }
    rows = reverse_int(rows);
    cols = reverse_int(cols);
    
    if (rows != 28 || cols != 28) {
        fprintf(stderr, "Error: Expected 28x28 images\n");
        fclose(file);
        return NULL;
    }
    
    size_t total_pixels = (size_t)(*count) * rows * cols;
    float *images = malloc(total_pixels * sizeof(float));
    unsigned char *buffer = malloc(total_pixels);
    
    if (!images || !buffer) {
        free(images);
        free(buffer);
        fclose(file);
        return NULL;
    }
    
    size_t read = fread(buffer, sizeof(unsigned char), total_pixels, file);
    if (read != total_pixels) {
        fprintf(stderr, "Error: Could not read all image data\n");
        free(images);
        free(buffer);
        fclose(file);
        return NULL;
    }
    
    // Normalize to [0, 1]
    for (size_t i = 0; i < total_pixels; i++) {
        images[i] = buffer[i] / 255.0f;
    }
    
    free(buffer);
    fclose(file);
    return images;
}

unsigned char* read_mnist_labels(const char *filename, int *count) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error: Cannot open %s\n", filename);
        return NULL;
    }
    
    int magic_number;
    if (fread(&magic_number, sizeof(int), 1, file) != 1) {
        fprintf(stderr, "Error: Failed to read magic number\n");
        fclose(file);
        return NULL;
    }
    magic_number = reverse_int(magic_number);
    
    if (magic_number != 2049) {
        fprintf(stderr, "Error: Invalid magic number in %s\n", filename);
        fclose(file);
        return NULL;
    }
    
    if (fread(count, sizeof(int), 1, file) != 1) {
        fprintf(stderr, "Error: Failed to read count\n");
        fclose(file);
        return NULL;
    }
    *count = reverse_int(*count);
    
    unsigned char *labels = malloc(*count);
    if (!labels) {
        fclose(file);
        return NULL;
    }
    
    size_t read = fread(labels, sizeof(unsigned char), *count, file);
    if (read != (size_t)*count) {
        fprintf(stderr, "Error: Could not read all label data\n");
        free(labels);
        fclose(file);
        return NULL;
    }
    
    fclose(file);
    return labels;
}

// Analysis functions
void analyze_predictions(Network *net, float *images, unsigned char *labels, int num_samples) {
    int confusion_matrix[10][10] = {0};
    int correct = 0;
    
    // Per-class statistics
    int class_total[10] = {0};
    int class_correct[10] = {0};
    
    clock_t start = clock();
    
    for (int i = 0; i < num_samples; i++) {
        // Forward pass
        forward_pass(net, &images[i * INPUT]);
        
        // Get prediction
        Layer *output_layer = &net->layers[net->num_layers - 1];
        int predicted = 0;
        float max_prob = output_layer->neurons[0];
        
        for (int j = 1; j < OUTPUT; j++) {
            if (output_layer->neurons[j] > max_prob) {
                max_prob = output_layer->neurons[j];
                predicted = j;
            }
        }
        
        int actual = labels[i];
        confusion_matrix[actual][predicted]++;
        class_total[actual]++;
        
        if (predicted == actual) {
            correct++;
            class_correct[actual]++;
        }
        
        // Progress indicator
        if ((i + 1) % 1000 == 0) {
            printf("\rProcessed %d/%d samples", i + 1, num_samples);
            fflush(stdout);
        }
    }
    
    double elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;
    printf("\rProcessed %d/%d samples - Done!\n\n", num_samples, num_samples);
    
    // Overall accuracy
    float overall_accuracy = 100.0f * correct / num_samples;
    printf("=== Overall Performance ===\n");
    printf("Total samples: %d\n", num_samples);
    printf("Correct predictions: %d\n", correct);
    printf("Incorrect predictions: %d\n", num_samples - correct);
    printf("Overall accuracy: %.2f%%\n", overall_accuracy);
    printf("Processing time: %.2f seconds (%.1f samples/sec)\n\n", 
           elapsed, num_samples / elapsed);
    
    // Per-class accuracy
    printf("=== Per-Class Performance ===\n");
    printf("Digit | Samples | Correct | Accuracy\n");
    printf("------|---------|---------|----------\n");
    for (int i = 0; i < 10; i++) {
        float class_accuracy = class_total[i] > 0 ? 
            100.0f * class_correct[i] / class_total[i] : 0.0f;
        printf("  %d   |  %5d  |  %5d  | %6.2f%%\n", 
               i, class_total[i], class_correct[i], class_accuracy);
    }
    
    // Confusion matrix
    printf("\n=== Confusion Matrix ===\n");
    printf("Rows: Actual, Columns: Predicted\n");
    printf("     ");
    for (int i = 0; i < 10; i++) {
        printf(" %4d", i);
    }
    printf("\n");
    printf("     ");
    for (int i = 0; i < 10; i++) {
        printf("-----");
    }
    printf("\n");
    
    for (int i = 0; i < 10; i++) {
        printf("  %d |", i);
        for (int j = 0; j < 10; j++) {
            if (confusion_matrix[i][j] > 0) {
                if (i == j) {
                    printf(" \033[32m%4d\033[0m", confusion_matrix[i][j]); // Green for correct
                } else {
                    printf(" \033[31m%4d\033[0m", confusion_matrix[i][j]); // Red for errors
                }
            } else {
                printf("    .");
            }
        }
        printf("\n");
    }
    
    // Find most confused pairs
    printf("\n=== Most Confused Pairs ===\n");
    typedef struct {
        int actual;
        int predicted;
        int count;
    } ConfusionPair;
    
    ConfusionPair pairs[90];  // 10*9 possible confusion pairs
    int pair_count = 0;
    
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            if (i != j && confusion_matrix[i][j] > 0) {
                pairs[pair_count].actual = i;
                pairs[pair_count].predicted = j;
                pairs[pair_count].count = confusion_matrix[i][j];
                pair_count++;
            }
        }
    }
    
    // Sort by count (simple bubble sort)
    for (int i = 0; i < pair_count - 1; i++) {
        for (int j = 0; j < pair_count - i - 1; j++) {
            if (pairs[j].count < pairs[j + 1].count) {
                ConfusionPair temp = pairs[j];
                pairs[j] = pairs[j + 1];
                pairs[j + 1] = temp;
            }
        }
    }
    
    printf("Actual -> Predicted (Count)\n");
    for (int i = 0; i < 5 && i < pair_count; i++) {
        printf("   %d   ->     %d     (%d times)\n", 
               pairs[i].actual, pairs[i].predicted, pairs[i].count);
    }
}

int main(void) {
    printf("=== MNIST Neural Network Test ===\n\n");
    
    // Load network
    printf("Loading neural network...\n");
    Network *net = load_network("weights.bin");
    if (!net) {
        fprintf(stderr, "Failed to load neural network!\n");
        fprintf(stderr, "Make sure you have trained the network first by running ./nn\n");
        return 1;
    }
    
    printf("Network architecture: %d", INPUT);
    for (int i = 0; i < net->num_layers; i++) {
        printf(" -> %d", net->layers[i].size);
    }
    printf("\n\n");
    
    // Load test data
    printf("Loading test data...\n");
    int num_images, num_labels;
    float *images = read_mnist_images("t10k-images-idx3-ubyte", &num_images);
    unsigned char *labels = read_mnist_labels("t10k-labels-idx1-ubyte", &num_labels);
    
    if (!images || !labels) {
        fprintf(stderr, "Failed to load test data!\n");
        free_network(net);
        free(images);
        free(labels);
        return 1;
    }
    
    if (num_images != num_labels) {
        fprintf(stderr, "Mismatch: %d images vs %d labels\n", num_images, num_labels);
        free_network(net);
        free(images);
        free(labels);
        return 1;
    }
    
    printf("Loaded %d test samples\n\n", num_images);
    
    // Analyze predictions
    analyze_predictions(net, images, labels, num_images);
    
    // Cleanup
    free_network(net);
    free(images);
    free(labels);
    
    return 0;
} 
