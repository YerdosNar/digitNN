#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

typedef struct {
    int size;
    int pre_size;
    float *neurons, *pre_train;
    float *biases, *bias_gradients;
    float **weights, **weight_gradients;
    float *deltas;
} Layer;

void init_layer(Layer *l, int pre_size, int size) {
    l->size = size;
    l->pre_size = pre_size;
    l->neurons = (float*)malloc(size * sizeof(float));
    l->pre_train = (float*)malloc(size * sizeof(float));
    l->biases = (float*)malloc(size * sizeof(float));
    l->bias_gradients = (float*)malloc(size * sizeof(float));
    l->weights = (float**)malloc(size * sizeof(float*));
    l->weight_gradients = (float**)malloc(size * sizeof(float*));
    l->deltas = (float*)calloc(size, sizeof(float));

    if(!l->neurons || !l->pre_train || !l->biases ||
       !l->bias_gradients || !l->weights || !l->weight_gradients ||
       !l->deltas) {
        printf("Memory allocation failed...\n");
exit(1);
    }

    for(int i = 0; i < size; i++) {
        l->weights[i] = (float*)malloc(pre_size * sizeof(float));
        l->weight_gradients[i] = (float*)calloc(pre_size, sizeof(float));
        if(!l->weights[i] || !l->weight_gradients[i]) {
            printf("Memory allocation failed... WEIGHTS\n");
            exit(1);
        }

        l->biases[i] = 0.0;
        for(int j = 0; j < pre_size; j++) {
            float limit = sqrt(6.0 / (pre_size + size));
            l->weights[i][j] = ((float)rand() / RAND_MAX) * 2 * limit - limit;
        }
    }
}

void init(Layer l[], int sizes[], int lay_num) {
    srand(time(NULL));
    for(int i = 0; i < lay_num; i++) {
        init_layer(&l[i], sizes[i], sizes[i+1]);
    }
}

void feedforward(float input[], int size, Layer *l, int relu) {
    if(!input || !l->neurons || !l->biases) {
        printf("Invalid layer data in feedforward.\n");
        return;
    }

    for(int i = 0; i < l->size; i++) {
        l->pre_train[i] = l->biases[i];
        for(int j = 0; j < size; j++) {
            l->pre_train[i] += input[j] * l->weights[i][j];
        }
        if(relu) {
            l->neurons[i] = fmaxf(0.0f, l->pre_train[i]);
} else {
            l->neurons[i] = l->pre_train[i];
        }
    }
}

void free_layer(Layer *l) {
    if(!l) return;

    for(int i = 0; i < l->size; i++) {
        free(l->weights[i]);
        free(l->weight_gradients[i]);
    }
    free(l->weights);
    free(l->weight_gradients);

    free(l->neurons);
    free(l->pre_train);
    free(l->biases);
    free(l->bias_gradients);
    free(l->deltas);
}

void free_all(Layer l[], int size) {
    for(int i = 0; i < size; i++) {
        free_layer(&l[i]);
    }
}

void shuffle(int *arr, int length) {
    for(int i = length-1; i > 0; i--) {
        int j = rand() % (i+1);
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
}

void softmax(float output[], int size) {
    float max = output[0];
    for(int i = 1; i < size; i++) {
        if(max < output[i]) {
            max = output[i];
        }
    }

    float sum = 0.0;
    for(int i = 0; i < size; i++) {
        output[i] = expf(output[i] - max);
        sum += output[i];
    }

    if(sum == 0) return;

    for(int i = 0; i < size; i++) {
        output[i] /= sum;
    }
}

void backpropagation(float input[], Layer **l, int lay_num, float target[]) {
    for(int i = 0; i < l[lay_num-1]->size; i++) {
        l[lay_num - 1]->deltas[i] = l[lay_num-1]->neurons[i] - target[i];
    }

    for(int k = lay_num - 2; k >= 0; k--) {
        for(int i = 0; i < l[k]->size; i++) {
            l[k]->deltas[i] = 0;
            for(int j = 0; j < l[k+1]->size; j++) {
                l[k]->deltas[i] += l[k+1]->deltas[j] * l[k+1]->weights[j][i];
            }
            l[k]->deltas[i] *= (l[k]->pre_train[i] > 0) ? 1.0f : 0.0f;
        }
    }

    for(int k = lay_num-1; k > 0; k--) {
        for(int i = 0; i < l[k]->size; i++) {
            l[k]->bias_gradients[i] += l[k]->deltas[i];
            for(int j = 0; j < l[k-1]->size; j++) {
                l[k]->weight_gradients[i][j] += l[k]->deltas[i] * l[k-1]->neurons[j];
            }
        }
    }

    for(int i = 0; i < l[0]->size; i++) {
        l[0]->bias_gradients[i] += l[0]->deltas[i];
        for(int j = 0; j < l[0]->pre_size; j++) {
            l[0]->weight_gradients[i][j] += l[0]->deltas[i] * input[j];
        }
    }
}

void update_parameters(Layer *l, float learn_rate, int batch_size) {
    for(int i = 0; i < l->size; i++) {
        l->biases[i] -= (learn_rate * l->bias_gradients[i]) / batch_size;
        for(int j = 0; j < l->pre_size; j++) {
            l->weights[i][j] -= (learn_rate * l->weight_gradients[i][j]) / batch_size;
            l->weight_gradients[i][j] = 0.0f;
        }
        l->bias_gradients[i] = 0.0f;
    }
}

void save_weights(Layer **l, int lay_num, const char* file) {
    FILE* f = fopen(file, "wb");

    for(int layer = 0; layer < lay_num; layer++) {
        fwrite(&l[layer]->size, sizeof(int), 1, f);
        fwrite(l[layer]->biases, sizeof(float), l[layer]->size, f);
        for(int i = 0; i < l[layer]->size; i++) {
            fwrite(l[layer]->weights[i], sizeof(float), l[layer]->pre_size, f);
        }
    }

    fclose(f);
}

int train(int lay_num, Layer *l, int lay_sizes[], int num_images, float* images, int num_labels, unsigned char* labels, int epochs, int batch_size, float learn_rate, const char *weights_bin_file) {
    init(l, lay_sizes, lay_num);
    int* indices = (int*)malloc(num_images * sizeof(int));
    for(int i = 0; i < num_images; i++) {
        indices[i] = i;
    }
    Layer **layers_ptr = (Layer**)malloc(lay_num * sizeof(Layer*));
    for(int i = 0; i < lay_num; i++) {
        layers_ptr[i] = &l[i];
    }

    double overall_time = 0.0;
    for(int epoch = 0; epoch < epochs; epoch++) {
        shuffle(indices, num_images);
        float total_loss = 0.0f;
        int correct = 0, incorrect = 0;

        clock_t start = clock();
        for(int batch = 0; batch < num_images/batch_size; batch++) {
            // RESETTING
            for(int k = 0; k < lay_num; k++) {
                memset(l[k].bias_gradients, 0, l[k].size * sizeof(float));
                for(int i = 0; i < l[k].size; i++) {
                    memset(l[k].weight_gradients[i], 0, l[k].pre_size * sizeof(float));
                }
            }

            for(int b = 0; b < batch_size; b++) {
                int idx = indices[batch * batch_size + b];
                float* image = &images[idx * lay_sizes[0]];
                float* target= (float*)calloc(lay_sizes[lay_num], sizeof(float));
                target[labels[idx]] = 1.0f;

                feedforward(image, lay_sizes[0], &l[0], 1);
                for(int lay = 0; lay < lay_num-2; lay++) {
                    feedforward(l[lay].neurons, lay_sizes[lay+1], &l[lay+1], 1);
                }
                feedforward(l[lay_num-2].neurons, lay_sizes[lay_num-1], &l[lay_num-1], 0);
                softmax(l[lay_num-1].neurons, lay_sizes[lay_num]);

                int max_idx = 0;
                for(int i = 1; i < lay_sizes[lay_num]; i++) {
                    if(l[lay_num-1].neurons[i] > l[lay_num-1].neurons[max_idx]) {
                        max_idx = i;
                    }
                }
                if(max_idx == labels[idx]) {
                    correct++;
                } else {
                    incorrect++;
                }

                for(int i = 0; i < lay_sizes[lay_num]; i++) {
                    total_loss += -target[i] * logf(l[lay_num-1].neurons[i] + 1e-8);
                }
                backpropagation(image, layers_ptr, lay_num, target);
                free(target);
            }
            for(int lay = 0; lay < lay_num; lay++) {
                update_parameters(&l[lay], learn_rate, batch_size);
            }
        }
        double time_spent = (double)(clock() - start) / CLOCKS_PER_SEC;
        overall_time += time_spent;
        printf("Epoch %d, Loss: %.3f, Accuracy: %.2f%%\nCorrect: %d, Incorrect: %d\n", epoch+1, total_loss/num_images, (float)correct/num_images*100, correct, incorrect);
        printf("TIME >>> %.4f seconds\n\n", time_spent);
    }
    save_weights(layers_ptr, lay_num, weights_bin_file);
    printf("Average epoch time: %.4f\n", overall_time/epochs);
    free(indices);
    free(layers_ptr);

    return 0;
}
