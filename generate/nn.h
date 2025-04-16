#ifndef NN_H
#define NN_H

#include <stdio.h>

typedef struct {
    int size;
    int pre_size;
    float *neurons, *pre_train;
    float *biases, *bias_gradients;
    float **weights, **weight_gradients;
    float *deltas;
} Layer;

void init_layer(Layer *l, int pre_size, int size);
void init(Layer l[], int size[], int lay_num);
void feedforward(float input[], int size, Layer *l, int  relu);
void free_layer(Layer *l);
void free_all(Layer l[], int size);
void shuffle(int *arr, int length);
void softmax(float output[], int size);
void backpropagation(float input[], Layer* l[], int lay_num, float target[]);
void update_parameters(Layer *l, float learn_rate, int batch_size);
void save_weights(Layer* l[], int lay_num, const char* file);
int train(int lay_num, Layer l[], int lay_sizes[], int num_images, float* images, int num_labels, unsigned char* labels, int epochs, int batch_size, float learn_rate, const char* weights_bin_file);

#endif
