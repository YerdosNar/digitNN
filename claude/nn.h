// nn.h - Neural Network Header File
// Contains all structure definitions and function declarations

#ifndef NN_H
#define NN_H

#include <stdbool.h>

typedef struct {
    int size;
    int pre_size;
    float *neurons;
    float *pre_activation;
    float *biases;
    float *bias_gradients;
    float *bias_momentum;
    float **weights;
    float **weight_gradients;
    float **weight_momentum;
    float *deltas;
} Layer;

typedef struct {
    Layer *layers;
    int num_layers;
    float learning_rate;
    float momentum;
    float l2_lambda;
} Network;

// Network creation and destruction
Network* create_network(int *layer_sizes, int num_layers);
void free_network(Network *net);

// Layer operations
bool init_layer(Layer *l, int pre_size, int size);
void free_layer(Layer *l);

// Forward propagation
void feedforward(const float *input, int input_size, Layer *l, bool use_relu);
void forward_pass(Network *net, const float *input);
void softmax_stable(float *output, int size);

// Backward propagation
void backward_pass(Network *net, const float *input, const float *target);
void update_parameters(Network *net, int batch_size);

// Training and validation
void train_network(Network *net, float *images, unsigned char *labels, 
                   int num_samples, int epochs, int batch_size);
float validate_network(Network *net, float *images, unsigned char *labels, int num_samples);

// I/O operations
float* read_mnist_images(const char *filename, int *count);
unsigned char* read_mnist_labels(const char *filename, int *count);
bool save_network(Network *net, const char *filename);

// Utility functions
void shuffle(int *arr, int n);
int reverse_int(int i);

#endif // NN_H
