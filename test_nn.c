#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define INPUT 784
#define HID1 25
#define HID2 25
#define OUTPUT 10

typedef struct {
    int size;
    int pre_size;
    float *neurons;
    float *biases;
    float **weights;
} Layer;

void free_layer(Layer *l) {
    if(!l) return;

    for(int i = 0; i < l->size; i++) {
        free(l->weights[i]);
    }
    free(l->weights);
    free(l->neurons);
    free(l->biases);
}

void free_all(Layer *l1, Layer *l2, Layer *l3) {
    free_layer(l1);
    free_layer(l2);
    free_layer(l3);
}

/* NN function */ 

void init_layer(Layer *l, int pre_size, int size) {
    l->size = size;
    l->pre_size = pre_size;
    l->neurons = (float*)calloc(size, sizeof(float));
    l->biases = (float*)calloc(size, sizeof(float));
    l->weights = (float**)malloc(size * sizeof(float*));
    if(!l->neurons || !l->biases || !l->weights) {
        printf("Memory allocation failed in init_layer\n");
        exit(1);
    }
    for(int i = 0; i < size; i++) {
        l->weights[i] = (float*)calloc(pre_size, sizeof(float));
        if(!l->weights[i]) {
            printf("Memory allocation failed for weights[%d].\n", i);
            exit(1);
        }
    }
}

void init(Layer *l1, Layer *l2, Layer *l3) {
    init_layer(l1, INPUT, HID1);
    init_layer(l2, HID1, HID2);
    init_layer(l3, HID2, OUTPUT);
}

void load_weights(Layer *l1, Layer *l2, Layer *l3, const char *name) {
    FILE *f = fopen(name, "rb");
    if(!f) {
        printf("Could NOT open \"%s\" file.\n", name);
        exit(1);
    }

    // First layer weights and biases
    int size;
    fread(&size, sizeof(int), 1, f);
    fread(l1->biases, sizeof(float), l1->size, f);
    for(int i = 0; i < size; i++) {
        fread(l1->weights[i], sizeof(float), l1->pre_size, f);
    }

    // Second Layer
    fread(&size, sizeof(int), 1, f);
    fread(l2->biases, sizeof(float), l2->size, f);
    for(int i = 0; i < size; i++) {
        fread(l2->weights[i], sizeof(float), l2->pre_size, f);
    }

    // Third lyaer
    fread(&size, sizeof(int), 1, f);
    fread(l3->biases, sizeof(float), l3->size, f);
    for(int i = 0; i < size; i++) {
        fread(l3->weights[i], sizeof(float), l3->pre_size, f);
    }

    printf("Weights loaded!\n");
    fclose(f);
}

void feedforward(float input[], int size, Layer *l, int activation_on) {
    if(!input || !l->neurons || !l->biases) {
        printf("Invalid layer data in feedforward.\n");
        return;
    }

    for(int i = 0; i < l->size; i++) {
        l->neurons[i] = l->biases[i];
        for(int j = 0; j < size; j++) {
            l->neurons[i] += input[j] * l->weights[i][j];
        }
        if(activation_on) {
            l->neurons[i] = fmaxf(0.0f, l->neurons[i]);
        }
    }
}

void softmax(float *output) {
    float max = output[0];
    for(int i = 1; i < OUTPUT; i++) {
        if(output[i] > max) max = output[i];
    }

    float sum = 0.0f;
    for(int i = 0; i < OUTPUT; i++) {
        output[i] = expf(output[i] - max);
        sum += output[i];
    }

    for(int i = 0; i < OUTPUT; i++) {
        output[i] /= sum;
    }
}

/* Reading functions */

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

    float* images = (float*)malloc(*count * INPUT * sizeof(float));
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

void run() {
    Layer l1, l2, l3;
    init(&l1, &l2, &l3);
    load_weights(&l1, &l2, &l3, "weights.bin");

    int num_images;
    float *input = read_mnist_images("t10k-images-idx3-ubyte", &num_images);
    int num_labels;
    unsigned char *labels = read_mnist_labels("t10k-labels-idx1-ubyte", &num_labels);

    if(num_images != num_labels) {
        printf("Mismatch! Number of images and labels are different!\n");
        free_all(&l1, &l2, &l3);
        free(input);
        free(labels);
        return;
    }

    int incorrect = 0;
    int correct = 0;
    for(int i = 0; i < num_images; i++) {
        feedforward(&input[i * INPUT], INPUT, &l1, 1);
        feedforward(l1.neurons, HID1, &l2, 1);
        feedforward(l2.neurons, HID2, &l3, 0);
        softmax(l3.neurons);

        int predicted = 0;
        float max_prob = l3.neurons[0];
        for(int j = 1; j < OUTPUT; j++) {
            if(l3.neurons[j] > max_prob) {
                max_prob = l3.neurons[j];
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
    free_all(&l1, &l2, &l3);
    free(input);
    free(labels);
}

int main() {
    run();

    return 0;
}
