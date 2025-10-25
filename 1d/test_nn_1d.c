#include "nn_1d.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void free_layer(Layer *l) {
    if(!l) return;

    free(l->weights);
    free(l->neurons);
    free(l->biases);
}

/* NN function */
void init_layer(Layer *l, int pre_size, int size) {
    l->size = size;
    l->pre_size = pre_size;
    l->neurons = (float*)calloc(size, sizeof(float));
    l->biases = (float*)calloc(size, sizeof(float));
    l->weights = (float*)malloc(size * pre_size * sizeof(float));
    if(!l->neurons || !l->biases || !l->weights) {
        printf("Memory allocation failed in init_layer\n");
        exit(1);
    }
}

void init(Layer l[], int sizes[], int lay_num) {
    for(int i = 0; i < lay_num; i++) {
        init_layer(&l[i], sizes[i], sizes[i+1]);
    }
}

Layer* load_model(const char* name, int* lay_num, int** lay_sizes_ptr) {
    FILE *f = fopen(name, "rb");
    if(!f) {
        printf("Could NOT open \"%s\" file.\n", name);
        return NULL;
    }

    // Read layer count
    if (fread(lay_num, sizeof(int), 1, f) != 1) {
        printf("Error reading layer count from %s\n", name);
        fclose(f);
        return NULL;
    }

    // Allocate and read layer sizes
    *lay_sizes_ptr = (int*)malloc((*lay_num + 1) * sizeof(int));
    if (fread(*lay_sizes_ptr, sizeof(int), *lay_num + 1, f) != (*lay_num + 1)) {
        printf("Error reading layer sizes from %s\n", name);
        fclose(f);
        free(*lay_sizes_ptr);
        return NULL;
    }

    // Allocate and initialize layers
    Layer *l = (Layer*)malloc(*lay_num * sizeof(Layer));
    init(l, *lay_sizes_ptr, *lay_num); // Uses the local init()

    // Read weights and biases
    for(int layer = 0; layer < *lay_num; layer++) {
        fread(l[layer].biases, sizeof(float), l[layer].size, f);
        fread(l[layer].weights, sizeof(float), l[layer].pre_size * l[layer].size, f);
    }

    fclose(f);
    printf("Model loaded from %s\n", name);
    return l;
}

void feedforward(float input[], int size, Layer *l, int activate) {
    if(!input || !l->neurons || !l->biases) {
        printf("Invalid layer in feedforward.\n");
        return;
    }

    for(int i = 0; i < l->size; i++) {
        l->neurons[i] = l->biases[i];
        for(int j = 0; j < size; j++) {
            l->neurons[i] += input[j] * l->weights[i * size + j];
        }
        if(activate) {
            l->neurons[i] = fmaxf(0.0f, l->neurons[i]);
        }
    }
}

void softmax(float output[], int length) {
    float max = output[0];
    for(int i = 1; i < length; i++)
        if(output[i] > max) max = output[i];

    float sum = 0.0;
    for(int i = 0; i < length; i++) {
        output[i] = exp(output[i] - max);
        sum += output[i];
    }

    for(int i = 0; i < length; i++)
        output[i] /= sum;
}

int reverse_int(int n) {
    unsigned char c1, c2, c3, c4;
    c1 = n & 255;
    c2 = (n >> 8) & 255;
    c3 = (n >> 16) & 255;
    c4 = (n >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + (int)c4;
}

float *read_mnist_images(const char* name, int* count) {
    FILE *f = fopen(name, "rb");
    if(!f) {
        printf("Could NOT open \"%s\" file.\n", name);
        exit(1);
    }

    int magic_number = 0;
    fread(&magic_number, sizeof(int), 1, f);
    magic_number = reverse_int(magic_number);

    fread(count, sizeof(int), 1, f);
    *count = reverse_int(*count);

    int rows, cols;
    fread(&rows, sizeof(int), 1, f);
    fread(&cols, sizeof(int), 1, f);
    rows = reverse_int(rows);
    cols = reverse_int(cols);

    float* images = (float*)malloc(*count * 784 * sizeof(float));
    unsigned char*buffer = (unsigned char*)malloc(*count * rows * cols);
    fread(buffer, sizeof(unsigned char), *count * rows * cols, f);

    for(int i = 0; i < *count * rows * cols; i++) {
        images[i] = buffer[i] / 255.0f;
    }

    fclose(f);
    free(buffer);
    return images;
}

unsigned char *read_mnist_labels(const char *name, int *count) {
    FILE *f = fopen(name, "rb");
    if(!f) {
        printf("Could NOT open \"%s\" file.\n", name);
        exit(1);
    }

    int magic_number;
    fread(&magic_number, sizeof(int), 1, f);
    magic_number = reverse_int(magic_number);

    fread(count, sizeof(int), 1, f);
    *count = reverse_int(*count);

    unsigned char *labels = (unsigned char*)malloc(*count);
    fread(labels, sizeof(unsigned char), *count, f);
    fclose(f);
    return labels;
}

void run(Layer *l, int lay_num, int lay_sizes[]) {
    int num_images;
    float *input = read_mnist_images("../files/t10k-images-idx3-ubyte", &num_images);
    int num_labels;
    unsigned char *labels = read_mnist_labels("../files/t10k-labels-idx1-ubyte", &num_labels);

    if(num_images != num_labels) {
        printf("Mismatch! Number of images and labels are different!\n");
        for(int i = 0; i < lay_num; i++) {
            free_layer(&l[i]);
        }
        free(input);
        free(labels);
        return;
    }

    int incorrect = 0;
    int correct = 0;
    for(int i = 0; i < num_images; i++) {
        float* curr_img = input + i * lay_sizes[0];
        feedforward(curr_img, lay_sizes[0], &l[0], 1);
        for(int lay = 0; lay < lay_num-2; lay++) {
            feedforward(l[lay].neurons, lay_sizes[lay+1], &l[lay+1], 1);
        }
        feedforward(l[lay_num-2].neurons, lay_sizes[lay_num-1], &l[lay_num-1], 0);
        softmax(l[lay_num-1].neurons, lay_sizes[lay_num]);

        int predicted = 0;
        float max_prob = l[lay_num-1].neurons[0];
        for(int j = 1; j < lay_sizes[lay_num]; j++) {
            if(l[lay_num-1].neurons[j] > max_prob) {
                max_prob = l[lay_num-1].neurons[j];
                predicted = j;
            }
        }

        if(predicted == labels[i]) {
            correct++;
        } else {
            incorrect++;
        }
    }

    printf("Accuracy: %.2f%% (%d %d)\nCorrect: %d\nIncorrect: %d\n",
            (correct / (float)num_images) * 100, correct, num_images, correct, incorrect);
    for(int i = 0; i < lay_num; i++) {
        free_layer(&l[i]);
    }
    free(input);
    free(labels);
}

int main() {
    int lay_num;
    int *lay_sizes;
    Layer *l = load_model("weights.bin", &lay_num, &lay_sizes);
    printf("Layer number: %d\n", lay_num);
    for(int i = 0; i < lay_num; i++) {
        printf("Layer %d: %d neurons\n", i, lay_sizes[i]);
    }

    if(!l) {
        return 1;
    }
    run(l, lay_num, lay_sizes);

    free(l);
    free(lay_sizes);
    return 0;
}
