#include "nn.h"
#include <stdio.h>
#include <stdlib.h>

int INPUT = 784;

int reverse_int(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

float* read_mnist_images(const char* filename, int* count) {
    FILE *file = fopen(filename, "rb");
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
    unsigned char* buffer = (unsigned char*) malloc(*count * rows * cols);
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

int main() {
    int num_images;
    float *images = read_mnist_images("train-images-idx3-ubyte", &num_images);
    int num_labels;
    unsigned char *labels = read_mnist_labels("train-labels-idx1-ubyte", &num_labels);

    // Creating Layers
    int lay_num;
    printf("Enter number of layers: ");
    scanf("%d", &lay_num);
    int lay_sizes[lay_num+1];
    lay_sizes[0] = INPUT;
    Layer layers[lay_num];
    for(int i = 1; i < lay_num; i++) {
        printf("Enter number of neurons in the Layer%d: ", i);
        scanf("%d", &lay_sizes[i]);
    }
    printf("Enter number of neurons in the OUTPUT Layer: ");
    scanf("%d", &lay_sizes[lay_num]);


    int epochs;
    printf("Enter number of epochs: ");
    scanf("%d", &epochs);

    // Training started
    int res = train(lay_num, layers, lay_sizes, num_images, images, num_labels, labels, epochs, 100, 0.01, "weights.bin");
    if(!res) {
        printf("Trainig is over!\nWeights saved to \"weights.bin\" file.\n");
        return 0;
    } else {
        return res;
    }

    return 0;
}
