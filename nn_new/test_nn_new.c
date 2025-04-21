#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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

void init(Layer l[], int sizes[], int lay_num) {
    for(int i = 0; i < lay_num; i++) {
        init_layer(&l[i], sizes[i], sizes[i+1]);
    }
}

void load_weights(Layer **l, int lay_num, const char* filename) {
    FILE* file = fopen(filename, "rb");
    if(!file) {
        printf("Could NOT open \"%s\" file\n", filename);
        exit(1);
    }

    for(int layer = 0; layer < lay_num; layer++) {
        int saved_size;
        fread(&saved_size, sizeof(int), 1, file);
        if(saved_size != l[layer]->size) {
            fprintf(stderr, "Layer %d size mismatch: expected %d, got %d\n", layer, l[layer]->size, saved_size);
        }
        fread(l[layer]->biases, sizeof(float), l[layer]->size, file);
        for(int i = 0; i < l[layer]->size; i++) {
            fread(l[layer]->weights[i], sizeof(float), l[layer]->pre_size, file);
        }
    }

    fclose(file);
}

void feedforward(float input[], int size, Layer *l, int activate) {
    if(!input || !l->neurons || !l->biases) {
        printf("Invalid layer in feedforward.\n");
        return;
    }

    for(int i = 0; i < l->size; i++) {
        l->neurons[i] = l->biases[i];
        for(int j = 0; j < size; j++) {
            l->neurons[i] += input[j] * l->weights[i][j];
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

/* Reading functions */

int reverse_int(int n) {
    unsigned char c1, c2, c3, c4;
    c1 = n & 255;
    c2 = (n >> 8) & 255;
    c3 = (n >> 16) & 255;
    c4 = (n >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + (int)c4;
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

    float* images = (float*)malloc(*count * 784 * sizeof(float));
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

void run(Layer *l, int lay_num, int lay_sizes[]) {
    Layer **lay_ptr = (Layer**)malloc(lay_num * sizeof(Layer*));
    for(int i = 0; i < lay_num; i++) {
        lay_ptr[i] = &l[i];
    }
    // Initialize neural network
    init(l, lay_sizes, lay_num);
    load_weights(lay_ptr, lay_num, "weights.bin");

    int num_images;
    float *input = read_mnist_images("t10k-images-idx3-ubyte", &num_images);
    int num_labels;
    unsigned char *labels = read_mnist_labels("t10k-labels-idx1-ubyte", &num_labels);

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
        float* curr_image = input + i * lay_sizes[0];
        feedforward(curr_image, lay_sizes[0], &l[0], 1);
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
    printf("Enter lay_num: ");
    scanf("%d", &lay_num);
    Layer *l = malloc(lay_num * sizeof(Layer));
    int lay_sizes[lay_num+1];
    lay_sizes[0] = 784;
    for(int i = 1; i < lay_num; i++) {
        printf("Enter Layer%d number of neurons: ", i);
        scanf("%d", &lay_sizes[i]);
    }
    printf("Enter OUTPUT layer number of neuroms: ");
    scanf("%d", &lay_sizes[lay_num]);

    run(l, lay_num, lay_sizes);
    return 0;
}
