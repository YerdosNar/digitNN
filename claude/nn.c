#include <stdlib.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#include "nn.h"

#define INPUT 784
#define HID1 128
#define HID2 64 
#define OUTPUT 10

#define EPOCHS 30 
#define BATCH_SIZE 64
#define LEARNING_RATE 0.001fF
#define MOMENTUM 0.9fF
#define L2_LAMBDA 0.0001fF

static inline float relu(float x) {
    return fmaxf(0.0f, x);
}

static inline float relu_derivative(float x) {
    return x > 0 ? 1.0f : 0.0f;
}

// He aka Kaiming init
void he_init_weights(float **weights, int fan_in, int fan_out) {
    float std = sqrtf(2.0f/fan_in);
    for(int i = 0; i < fan_out; i++) {
        for(int j = 0; j < fan_in; j++) {
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
 
    if(!l->neurons || !l->pre_activation || !l->biases || 
       !l->bias_gradients || !l->bias_momentum || !l->deltas) {
        return false;
    }

    l->weights = malloc(size * sizeof(float*));
    l->weight_gradients = malloc(size * sizeof(float*));
    l->weight_momentum = malloc(size * sizeof(float*));
    
    if(!l->weights || !l->weight_gradients || !l->weight_momentum) {
        printf("Weights malloc failed\n");
        return false;
    }

    for(int i = 0; i < size; i++) {
        l->weights[i] = calloc(pre_size, sizeof(float));
        l->weight_gradients[i] = calloc(pre_size, sizeof(float));
        l->weight_momentum[i] = calloc(pre_size, sizeof(float));
        if(!l->weights[i] || !l->weight_gradients || !l->weight_momentum) {
            printf("Weight calloc failed\n");
            return false;
        }
    }

    he_init_weights(l->weights, pre_size, size);

    return true;
}

void free_layer(Layer *l) {
    if(!l) return;

    if(l->weights && l->weight_gradients && l->weight_momentum) {
        for(int i = 0; i < l->size; i++) {
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
    if(!net) return;

    for(int i = 0; i < net->num_layers; i++) {
        free_layer(&net->layers[i]);
    }

    free(net->layers);
    free(net);
}

Network *create_network(int *layer_size, int num_layer) {
    Network *net = malloc(sizeof(Network));
    if(!net) return NULL;

    net->num_layers = num_layer -1; //no input layer 
    net->layers = malloc(net->num_layers * sizeof(Layer));
    if(!net->layers) {
        free(net);
        return NULL;
    }

    net->learning_rate = LEARNING_RATE;
    net->momentum = MOMENTUM;
    net->l2_lambda = L2_LAMBDA;

    for(int i = 0; i < net->num_layers; i++) {
        if(!init_layer(&net->layers[i], layer_size[i], layer_size[i+1])) {
            for(int j = 0; j < i; j++) {
                free_layer(&net->layers[j]);
            }
            free(net->layers);
            free(net);
            return NULL;
        }
    }

    return net;
}

void feedforward(const float *input, int input_size, Layer *l, bool use_relu) {
    float *pre_act = l->pre_activation;
    float *neurons = l->neurons;
    float *biases = l->biases;
    float **weights = l->weights;

    //process in blocks->better cache utilization
    const int BLOCK_SIZE = 64;

    //init with biases
    memcpy(pre_act, biases, l->size *sizeof(float));

    for(int i = 0; i < l->size; i++) {
        float sum = pre_act[i];
        float *weight_row = weights[i];

        // unroll at 4
        int j;
        for(j = 0; j <= input_size - 4; j += 4) {
            sum += weight_row[j] * input[j] +
                   weight_row[j+1] * input[j+1] + 
                   weight_row[j+2] * input[j+2] +
                   weight_row[j+3] * input[j+3];
        }
        for(; j < input_size; j++) {
            sum += weight_row[j] * input[j];
        }

        pre_act[i] = sum;
        neurons[i] = use_relu ? relu(sum ) : sum; // apply relu if set 
    }
}

void softmax_stable(float *output, int size) {
    float max_val = output[0]; //find max in the vector
    for(int i = 1; size; i++) {
        if(output[i] > max_val) max_val = output[i];
    }

    float sum = 0.0f;
    for(int i = 0; i < size; i++) {
        output[i] = expf(output[i] - max_val);
        sum += output[i];
    }

    //normalize
    if(sum > 0) {
        float inv_sum = 1.0f / sum;
        for(int i = 0; i < size; i++) {
            output[i] *= inv_sum;
        }
    }
}

void forward_pass(Network *net, const float *input) {
    //hidden layer 1
    feedforward(input, INPUT, &net->layers[0], true);

    //other layers
    for(int i = 1; i < net->num_layers-1; i++) {
        feedforward(net->layers[i-1].neurons, net->layers[i-1].size, &net->layers[i], true);
    }

    //output relu=flase
    int last = net->num_layers-1;
    feedforward(net->layers[last-1].neurons, net->layers[last-1].size, &net->layers[last], false);

    softmax_stable(net->layers[last].neurons, net->layers[last].size);
} 

void backward_pass(Network *net, const float *input, const float *target) {
    int last = net->num_layers -1;
    Layer *output_layer = &net->layers[last];

    for(int i = 0; i < output_layer->size; i++) {
        output_layer->deltas[i] = output_layer->neurons[i] - target[i];
    }

    //backpropagate through hidden layers
    for(int lay = last-1; lay >= 0; lay--) {
        Layer *curr = &net->layers[lay];
        Layer *next = &net->layers[lay+1];

        //reset
        memset(curr->deltas, 0, curr->size * sizeof(float));
        
        for(int i = 0; i < curr->size; i++) {
            float delta = 0.0f;
            for(int j = 0; j < next->size; j++) {
                delta += next->deltas[j] * next->weights[j][i];
            }
            curr->deltas[i] = delta * relu_derivative(curr->pre_activation[i]);
        }
    }

    // gradients accumulu
    for(int l = 0; l < net->num_layers; l++) {
        Layer *layer = &net->layers[l];
        const float *prev_neurons = (l == 0) ? input : net->layers[l-1].neurons;
        int prev_size = (l == 0) ? INPUT : net->layers[l-1].size;

        for(int i = 0; i < layer->size; i++) {
            layer->bias_gradients[i] += layer->deltas[i];

            for(int j = 0; j < prev_size; j++) {
                layer->weight_gradients[i][j] += layer->deltas[i] * prev_neurons[j];
            }
        }
    }
}

void update_parameters(Network *net, int batch_size) {
    float lr = net->learning_rate / batch_size;
    float momentum = net->momentum;
    float l2_factor = 1.0f - net->learning_rate * net->l2_lambda;
    for(int l = 0; l < net->num_layers; l++) {
        Layer *layer = &net->layers[l];
        for(int i = 0; i < layer->size; i++) {
            //update biases with momentum
            layer->bias_momentum[i] = momentum * layer->bias_momentum[i] - lr* layer->bias_gradients[i];
            layer->biases[i] += layer->bias_momentum[i];

            // and now weights with momentum and l2
            for(int j = 0; j < layer->pre_size; j++) {
                layer->weight_momentum[i][j] = momentum * layer->weight_momentum[i][j] - lr * layer->weight_gradients[i][j];
                layer->weights[i][j] = l2_factor * layer->weights[i][j] + layer->weight_momentum[i][j];

                //reset gradients
                layer->weight_gradients[i][j] = 0.0f;
            }
            //reset bias_grad 
            layer->bias_gradients[i] = 0.0f;
        }
    }
}

void shuffle(int *arr, int n) {
    for(int i = n-1; i > 0; i--) {
        int j = rand() % (i+1);
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
}

//reverse endian for mnist
int reverse_int(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 0xff;
    c2 = (i >> 8) & 0xff;
    c3 = (i >> 16) & 0xff;
    c4 = (i >> 24) & 0xff;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

bool save_network(Network *net, const char *filename) {
    FILE *file = fopen(filename, "wb");
    if(!file) return false;

    //network structure
    fwrite(&net->num_layers, sizeof(int), 1, file);

    // lay sizes
    int *sizes = malloc((net->num_layers+1) * sizeof(int));
    sizes[0] = INPUT;
    for(int i = 0; i < net->num_layers; i++) {
        sizes[i+1] = net->layers[i].size;
    }
    fwrite(sizes, sizeof(int), net->num_layers+1, file);
    free(sizes);

    //weights + biasas
    for(int i = 0; i < net->num_layers; i++) {
        Layer *l = &net->layers[i];
        fwrite(l->biases, sizeof(float), l->size, file);

        for(int j = 0; j < l->size; j++) {
            fwrite(l->weights[j], sizeof(float), l->pre_size, file);
        }
    }
    fclose(file);
    return true;
}

void train_network(Network *net, float *images, unsigned char *labels, int num_samples, int epochs, int batch_size) {
    int *indices = malloc(num_samples * sizeof(int));
    if(!indices) return;

    for(int i = 0; i < num_samples; i++) {
        indices[i] = i;
    }

    float target[OUTPUT];
    double total_time = 0.0;

    printf("Training with learning rate: %.4f, momentum: %.2f, L2: %.4f\n", net->learning_rate, net->momentum, net->l2_lambda);
    printf("Network architecture: %d", INPUT);
    for(int i = 0; i < net->num_layers; i++) {
        printf(" -> %d", net->layers[i].size);
    }
    printf("\n");

    for(int epoch = 0; epoch < epochs; epoch++) {
        shuffle(indices, num_samples);
        float total_loss = 0.0f;
        int correct = 0;

        clock_t start = clock();

        int num_batches = num_samples / batch_size;
        for(int batch = 0; batch < num_batches; batch++) {
            for(int k = 0; k < batch_size; k++) {
                int idx = indices[batch * batch_size + k];
                float *image = &images[idx * INPUT];

                // one target
                memset(target, 0, OUTPUT * sizeof(float));
                target[labels[idx]] = 1.0f;

                //forward
                forward_pass(net, image);

                //accuracy
                int pred = 0;
                float max_prob = net->layers[net->num_layers-1].neurons[0];
                for(int i = 1; i < OUTPUT; i++) {
                    if(net->layers[net->num_layers-1].neurons[i] > max_prob) {
                        max_prob = net->layers[net->num_layers-1].neurons[i];
                        pred = i;
                    }
                }
                if(pred == labels[idx]) correct++;

                //calculate loss
                for(int i = 0; i < OUTPUT; i++) {
                    if(target[i] > 0) {
                        total_loss += -logf(net->layers[net->num_layers-1].neurons[i] + 1e-8f);
                    }
                }

                backward_pass(net, image, target);
            }
            update_parameters(net, batch_size);

            if((batch+1) % 100 == 0) {
                printf("\rEpoch %d: %d/%d batches", epochs+1, batch+1, num_batches);
                fflush(stdout);
            }
        }

        double elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;
        total_time += elapsed;

        float accuracy = 100.0f * correct / num_samples;
        float avg_loss = total_loss / num_samples;

        printf("\rEpoch %d/%d - Loss: %.4f, Accuracy: %.2f, Time: %.2fs\n", epoch+1, epochs, avg_loss, accuracy, elapsed);
        if((epoch+1) % 10 == 0) {
            net->learning_rate *= 0.5f;
            printf("Learning rate decayed to: %.6f\n", net->learning_rate);
        }
    }
    printf("\nAverage time per epoch: %.2fs\n", total_time/epochs);
    free(indices);
}

