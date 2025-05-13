#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#define INPUT 784
#define HID1 16
#define HID2 16
#define OUTPUT 10

#define EPOCHS 50
#define BATCH_SIZE 100
#define LEARNING_RATE 0.01

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

void init(Layer *l1, Layer *l2, Layer *l3) {
    srand(time(NULL));
    init_layer(l1, INPUT, HID1);
    init_layer(l2, HID1, HID2);
    init_layer(l3, HID2, OUTPUT);
}

void shuffle(int *arr, int length) {
    for(int i = length-1; i > 0; i--) {
        int j = rand() % (i+1);
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
}

void feedforward(float input[], int size, Layer *l, int activation_func) {
    if(!input || !l->neurons || !l->biases) {
        printf("Invalid layer data in feedforward.\n");
        return;
    }

    for(int i = 0; i < l->size; i++) {
        l->pre_train[i] = l->biases[i];
        for(int j = 0; j < size; j++) {
            l->pre_train[i] += input[j] * l->weights[i][j];
        }
        if(activation_func) {
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

void free_all(Layer *l1, Layer *l2, Layer *l3) {
    free_layer(l1);
    free_layer(l2);
    free_layer(l3);
}

void softmax(float output[]) {
    float max = output[0];
    for(int i = 1; i < OUTPUT; i++) {
        if(max < output[i]) max = output[i];
    }

    float sum = 0.0;
    for(int i = 0; i < OUTPUT; i++) {
        output[i] = expf(output[i] - max);
        sum += output[i];
    }
    if(sum == 0) return;

    for(int i = 0; i < OUTPUT; i++) {
        output[i] /= sum;
    }
}

void backpropagation(float input[], Layer *l1, Layer *l2, Layer *l3, float target[], float learn_rate) {
    for(int i = 0; i < l3->size; i++) {
        l3->deltas[i] = l3->neurons[i] - target[i];
    }

    //Hidden layer 2
    for(int i = 0; i < l2->size; i++) {
        l2->deltas[i] = 0;
        for(int j = 0; j < l3->size; j++) {
            l2->deltas[i] += l3->deltas[j] * l3->weights[j][i];
        }
        l2->deltas[i] *= (l2->pre_train[i] > 0) ? 1.0f : 0.0f;
    }

    // Hidden layer 1
    for(int i = 0; i < l1->size; i++) {
        l1->deltas[i] = 0;
        for(int j = 0; j < l2->size; j++) {
            l1->deltas[i] += l2->deltas[j] * l2->weights[j][i];
        }
        l1->deltas[i] *= (l1->pre_train[i] > 0) ? 1.0f : 0.0f;
    }

    // Output layer, updating gradients
    for(int i = 0; i < l3->size; i++) {
        l3->bias_gradients[i] += l3->deltas[i];
        for(int j = 0; j < l2->size; j++) {
            l3->weight_gradients[i][j] += l3->deltas[i] * l2->neurons[j];
        }
    }

    // Hidden layer 2
    for(int i = 0; i < l2->size; i++) {
        l2->bias_gradients[i] += l2->deltas[i];
        for(int j = 0; j < l1->size; j++) {
            l2->weight_gradients[i][j] += l2->deltas[i] * l1->neurons[j];
        }
    }

    // Hidden layer 1
    for(int i = 0; i < l1->size; i++) {
        l1->bias_gradients[i] += l1->deltas[i];
        for(int j = 0; j < INPUT; j++) {
            l1->weight_gradients[i][j] += l1->deltas[i] * input[j];
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

int reverse_int(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

float* read_mnist_images(const char* filename, int* count) {
    FILE* file = fopen(filename, "rb");
    if(!file) {
        printf("Could NOT open \"%s\" file.\n", filename);
        exit(1);
    }

    int magic_number = 0;
    fread(&magic_number, sizeof(int), 1, file);
    magic_number = reverse_int(magic_number);

    fread(count, sizeof(int), 1, file);
    *count = reverse_int(*count);

    int rows, cols;
    fread(&rows, sizeof(int), 1, file);
    fread(&cols, sizeof(int), 1, file);
    rows = reverse_int(rows);
    cols = reverse_int(cols);

    float* images = (float*)malloc(*count * INPUT * sizeof(float));
    unsigned char* buffer = (unsigned char*)malloc(*count * rows * cols);
    fread(buffer, sizeof(unsigned char), *count * rows * cols, file);

    for(int i = 0; i < *count * rows * cols; i++) {
        images[i] = buffer[i] / 255.0f;
    }

    fclose(file);
    free(buffer);
    return images;
}

unsigned char* read_mnist_labels(const char* filename, int* count) {
    FILE* file = fopen(filename, "rb");
    if(!file) {
        printf("Could NOT open \"%s\" file.\n", filename);
        exit(1);
    }

    int magic_number;
    fread(&magic_number, sizeof(int), 1, file);
    magic_number = reverse_int(magic_number);

    fread(count, sizeof(int), 1, file);
    *count = reverse_int(*count);

    unsigned char* labels = (unsigned char*)malloc(*count);
    fread(labels, sizeof(unsigned char), *count, file);
    fclose(file);
    return labels;
}

void save_weights(Layer *l1, Layer *l2, Layer *l3, const char* filename) {
    FILE* file = fopen(filename, "wb");

    fwrite(&l1->size, sizeof(int), 1, file);
    fwrite(l1->biases, sizeof(float), l1->size, file);
    for(int i = 0; i < l1->size; i++) {
        fwrite(l1->weights[i], sizeof(float), l1->pre_size, file);
    }

    fwrite(&l2->size, sizeof(int), 1, file);
    fwrite(l2->biases, sizeof(float), l2->size, file);
    for(int i = 0; i < l2->size; i++) {
        fwrite(l2->weights[i], sizeof(float), l2->pre_size, file);
    }

    fwrite(&l3->size, sizeof(int), 1, file);
    fwrite(l3->biases, sizeof(float), l3->size, file);
    for(int i = 0; i < l3->size; i++) {
        fwrite(l3->weights[i], sizeof(float), l3->pre_size, file);
    }
    fclose(file);
}

int run() {
    Layer l1, l2, l3;
    init(&l1, &l2, &l3);

    int num_images;
    float* images = read_mnist_images("train-images-idx3-ubyte", &num_images);
    int num_labels;
    unsigned char* labels = read_mnist_labels("train-labels-idx1-ubyte", &num_labels);

    if(num_images != num_labels) {
        printf("Mismatch! Number of images and labels are different!\n");
        return 1;
    }

    int* indices = (int*)malloc(num_images * sizeof(int));
    for(int i = 0; i < num_images; i++) indices[i] = i;

    double overall_time = 0.0;
    for(int epoch = 0; epoch < EPOCHS; epoch++) {
        shuffle(indices, num_images);
        float total_loss = 0.0f;
        int correct = 0, incorrect = 0;

        clock_t start = clock();
        for(int batch = 0; batch < num_images/BATCH_SIZE; batch++) {
            memset(l1.bias_gradients, 0, HID1 * sizeof(float));
            memset(l2.bias_gradients, 0, HID2 * sizeof(float));
            memset(l3.bias_gradients, 0, OUTPUT * sizeof(float));

            for(int i = 0; i < HID1; i++) memset(l1.weight_gradients[i], 0, INPUT * sizeof(float));
            for(int i = 0; i < HID2; i++) memset(l2.weight_gradients[i], 0, HID1 * sizeof(float));
            for(int i = 0; i < OUTPUT; i++) memset(l3.weight_gradients[i], 0, HID2 * sizeof(float));

            for(int k = 0; k < BATCH_SIZE; k++) {
                int idx = indices[batch * BATCH_SIZE + k];
                float* image = &images[idx * INPUT];
                float target[OUTPUT] = {0.0f};
                target[labels[idx]] = 1.0f;

                feedforward(image, INPUT, &l1, 1);
                feedforward(l1.neurons, HID1, &l2, 1);
                feedforward(l2.neurons, HID2, &l3, 0);
                softmax(l3.neurons);

                int max_idx = 0;
                for(int i = 1; i < OUTPUT; i++) {
                    if(l3.neurons[i] > l3.neurons[max_idx]) max_idx = i;
                }
                if(max_idx == labels[idx]) {
                    correct++;
                } else {
                    incorrect++;
                }

                for(int i = 0; i < OUTPUT; i++) {
                    total_loss += -target[i] * logf(l3.neurons[i] + 1e-8);
                }

                backpropagation(image, &l1, &l2, &l3, target, LEARNING_RATE);
            }
            update_parameters(&l1, LEARNING_RATE, BATCH_SIZE);
            update_parameters(&l2, LEARNING_RATE, BATCH_SIZE);
            update_parameters(&l3, LEARNING_RATE, BATCH_SIZE);
        }
        double time_spent = (double)(clock() - start) / CLOCKS_PER_SEC;
        overall_time += time_spent;
        printf("Epoch %d, Loss: %.3f, Accuracy: %.2f%%\nCorrect: %d, Incorrect: %d\n", epoch+1, total_loss/num_images, (float)correct/num_images*100, correct, incorrect);
        printf("TIME >>> %.3f\n\n", time_spent);
    }
    save_weights(&l1, &l2, &l3, "weights.bin");
    printf("Average epoch time: %.3f\n", overall_time/EPOCHS);

    free(images);
    free(labels);
    free(indices);
    free_all(&l1, &l2, &l3);

    return 0;
}

int main() {
    int result = run();
    if(result == 0) {
        printf("SUCCESS!\nWeights saved to \"weights.bin\" file.\n");
        return 0;
    } else {
        return result;
    }
}
