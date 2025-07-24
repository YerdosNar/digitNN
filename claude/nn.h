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

//network init and free
Network *create_network(int *layer_size, int num_layers);
bool init_layer(Layer *l, int pre_size, int size);
void free_network(Network *net);

//forward
void feedforward(const float *input, int input_size, Layer *l, bool use_relu);
void forward_pass(Network *net, const float *input);
void softmax_stable(float *output, int size);

//backward
void backward_pass(Network *net, const float *input, const float *target);
void update_parameters(Network *net, int batch_size);

//train 
void train_network(Network *net, float *images, unsigned char *labels, int num_samples, int eposhc, int batch_size);
float validate_network(Network *net, float *images, unsigned char *labels, int num_samples);

// I/O
float *read_mnist_images(const char *filename, int *count);
unsigned char *read_mnist_labels(const char *filename, int *count);
bool save_network(Network *net, const char *filename);
Network *load_network(const char *filename);

// utility functions
void shuffle(int *arr, int n);
int reverse_int(int i);

// activation functions
static inline float relu(float x);
static inline float relu_derivative(float x);

#endif
