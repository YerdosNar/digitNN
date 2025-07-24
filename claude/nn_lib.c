// nn_lib.c - Shared Neural Network Library
// This file contains all the common functions used by nn.c, test_nn.c, and draw_test.c
// Compile to object file: gcc -c nn_lib.c -o nn_lib.o -O3

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>

#include "nn.h"

#define INPUT 784
#define OUTPUT 10

// Activation functions
float relu(float x) {
    return fmaxf(0.0f, x);
}

float relu_derivative(float x) {
    return x > 0 ? 1.0f : 0.0f;
}

// He initialization for ReLU
void he_init_weights(float **weights, int fan_in, int fan_out) {
    float std = sqrtf(2.0f / fan_in);
    for (int i = 0; i < fan_out; i++) {
        for (int j = 0; j < fan_in; j++) {
            // Box-Muller transform for normal distribution
            float u1 = (float)rand() / RAND_MAX;
            float u2 = (float)rand() / RAND_MAX;
            float z0 = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
            weights[i][j] = z0 * std;
        }
    }
}

bool init_layer(Layer *l, int pre_size, int size) {
    l->size = size;
    l->pre_size = pre_size;
    
    l->neurons = calloc(size, sizeof(float));
    l->pre_activation = calloc(size, sizeof(float));
    l->biases = calloc(size, sizeof(float));
    l->bias_gradients = calloc(size, sizeof(float));
    l->bias_momentum = calloc(size, sizeof(float));
    l->deltas = calloc(size, sizeof(float));
    
    if (!l->neurons || !l->pre_activation || !l->biases || 
        !l->bias_gradients || !l->bias_momentum || !l->deltas) {
        return false;
    }
    
    l->weights = malloc(size * sizeof(float*));
    l->weight_gradients = malloc(size * sizeof(float*));
    l->weight_momentum = malloc(size * sizeof(float*));
    
    if (!l->weights || !l->weight_gradients || !l->weight_momentum) {
        return false;
    }
    
    for (int i = 0; i < size; i++) {
        l->weights[i] = calloc(pre_size, sizeof(float));
        l->weight_gradients[i] = calloc(pre_size, sizeof(float));
        l->weight_momentum[i] = calloc(pre_size, sizeof(float));
        
        if (!l->weights[i] || !l->weight_gradients[i] || !l->weight_momentum[i]) {
            return false;
        }
    }
    
    return true;
}

void free_layer(Layer *l) {
    if (!l) return;
    
    if (l->weights && l->weight_gradients && l->weight_momentum) {
        for (int i = 0; i < l->size; i++) {
            free(l->weights[i]);
            free(l->weight_gradients[i]);
            free(l->weight_momentum[i]);
        }
    }
    
    free(l->weights);
    free(l->weight_gradients);
    free(l->weight_momentum);
    free(l->neurons);
    free(l->pre_activation);
    free(l->biases);
    free(l->bias_gradients);
    free(l->bias_momentum);
    free(l->deltas);
}

void free_network(Network *net) {
    if (!net) return;
    
    for (int i = 0; i < net->num_layers; i++) {
        free_layer(&net->layers[i]);
    }
    free(net->layers);
    free(net);
}

// Optimized matrix multiplication with cache-friendly access
void feedforward(const float *input, int input_size, Layer *l, bool use_relu) {
    float *pre_act = l->pre_activation;
    float *neurons = l->neurons;
    float *biases = l->biases;
    float **weights = l->weights;
    
    // Initialize with biases
    memcpy(pre_act, biases, l->size * sizeof(float));
    
    // Blocked matrix multiplication
    for (int i = 0; i < l->size; i++) {
        float sum = pre_act[i];
        float *weight_row = weights[i];
        
        // Unroll loop for better performance
        int j;
        for (j = 0; j <= input_size - 4; j += 4) {
            sum += weight_row[j] * input[j] +
                   weight_row[j+1] * input[j+1] +
                   weight_row[j+2] * input[j+2] +
                   weight_row[j+3] * input[j+3];
        }
        
        // Handle remaining elements
        for (; j < input_size; j++) {
            sum += weight_row[j] * input[j];
        }
        
        pre_act[i] = sum;
        neurons[i] = use_relu ? relu(sum) : sum;
    }
}

void softmax_stable(float *output, int size) {
    // Find max for numerical stability
    float max_val = output[0];
    for (int i = 1; i < size; i++) {
        if (output[i] > max_val) max_val = output[i];
    }
    
    // Compute exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        output[i] = expf(output[i] - max_val);
        sum += output[i];
    }
    
    // Normalize
    if (sum > 0) {
        float inv_sum = 1.0f / sum;
        for (int i = 0; i < size; i++) {
            output[i] *= inv_sum;
        }
    }
}

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

// Fisher-Yates shuffle
void shuffle(int *arr, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
}

// Reverse endianness for MNIST files
int reverse_int(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

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

bool save_network(Network *net, const char *filename) {
    FILE *file = fopen(filename, "wb");
    if (!file) return false;
    
    // Write network structure
    fwrite(&net->num_layers, sizeof(int), 1, file);
    
    // Write layer sizes
    int *sizes = malloc((net->num_layers + 1) * sizeof(int));
    sizes[0] = INPUT;
    for (int i = 0; i < net->num_layers; i++) {
        sizes[i + 1] = net->layers[i].size;
    }
    fwrite(sizes, sizeof(int), net->num_layers + 1, file);
    free(sizes);
    
    // Write weights and biases
    for (int i = 0; i < net->num_layers; i++) {
        Layer *l = &net->layers[i];
        fwrite(l->biases, sizeof(float), l->size, file);
        
        for (int j = 0; j < l->size; j++) {
            fwrite(l->weights[j], sizeof(float), l->pre_size, file);
        }
    }
    
    fclose(file);
    return true;
}

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
