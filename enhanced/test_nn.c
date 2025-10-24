// test_nn.c - Test Program for Evaluating the Trained Neural Network
// Compile: gcc test_nn.c nn_lib.c -o test_nn -lm -O3 -march=native
// Run after training: ./test_nn

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "nn_lib.h"

// Analysis functions
void analyze_predictions(Network *net, float *images, unsigned char *labels, int num_samples) {
    int confusion_matrix[10][10] = {0};
    int correct = 0;

    // Per-class statistics
    int class_total[10] = {0};
    int class_correct[10] = {0};

    clock_t start = clock();

    for (int i = 0; i < num_samples; i++) {
        // Forward pass
        forward_pass(net, &images[i * INPUT_SIZE]);

        // Get prediction
        Layer *output_layer = &net->layers[net->num_layers - 1];
        int predicted = 0;
        float max_prob = output_layer->neurons[0];

        for (int j = 1; j < OUTPUT_SIZE; j++) {
            if (output_layer->neurons[j] > max_prob) {
                max_prob = output_layer->neurons[j];
                predicted = j;
            }
        }

        int actual = labels[i];
        confusion_matrix[actual][predicted]++;
        class_total[actual]++;

        if (predicted == actual) {
            correct++;
            class_correct[actual]++;
        }

        // Progress indicator
        if ((i + 1) % 1000 == 0) {
            printf("\rProcessed %d/%d samples", i + 1, num_samples);
            fflush(stdout);
        }
    }

    double elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;
    printf("\rProcessed %d/%d samples - Done!\n\n", num_samples, num_samples);

    // Overall accuracy
    float overall_accuracy = 100.0f * correct / num_samples;
    printf("=== Overall Performance ===\n");
    printf("Total samples: %d\n", num_samples);
    printf("Correct predictions: %d\n", correct);
    printf("Incorrect predictions: %d\n", num_samples - correct);
    printf("Overall accuracy: %.2f%%\n", overall_accuracy);
    printf("Processing time: %.2f seconds (%.1f samples/sec)\n\n",
           elapsed, num_samples / elapsed);

    // Per-class accuracy
    printf("=== Per-Class Performance ===\n");
    printf("Digit | Samples | Correct | Accuracy\n");
    printf("------|---------|---------|----------\n");
    for (int i = 0; i < 10; i++) {
        float class_accuracy = class_total[i] > 0 ?
            100.0f * class_correct[i] / class_total[i] : 0.0f;
        printf("  %d   |  %5d  |  %5d  | %6.2f%%\n",
               i, class_total[i], class_correct[i], class_accuracy);
    }

    // Confusion matrix
    printf("\n=== Confusion Matrix ===\n");
    printf("Rows: Actual, Columns: Predicted\n");
    printf("     ");
    for (int i = 0; i < 10; i++) {
        printf(" %4d", i);
    }
    printf("\n");
    printf("     ");
    for (int i = 0; i < 10; i++) {
        printf("-----");
    }
    printf("\n");

    for (int i = 0; i < 10; i++) {
        printf("  %d |", i);
        for (int j = 0; j < 10; j++) {
            if (confusion_matrix[i][j] > 0) {
                if (i == j) {
                    printf(" \033[32m%4d\033[0m", confusion_matrix[i][j]); // Green for correct
                } else {
                    printf(" \033[31m%4d\033[0m", confusion_matrix[i][j]); // Red for errors
                }
            } else {
                printf("    .");
            }
        }
        printf("\n");
    }

    // Find most confused pairs
    printf("\n=== Most Confused Pairs ===\n");
    typedef struct {
        int actual;
        int predicted;
        int count;
    } ConfusionPair;

    ConfusionPair pairs[90];  // 10*9 possible confusion pairs
    int pair_count = 0;

    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            if (i != j && confusion_matrix[i][j] > 0) {
                pairs[pair_count].actual = i;
                pairs[pair_count].predicted = j;
                pairs[pair_count].count = confusion_matrix[i][j];
                pair_count++;
            }
        }
    }

    // Sort by count (simple bubble sort)
    for (int i = 0; i < pair_count - 1; i++) {
        for (int j = 0; j < pair_count - i - 1; j++) {
            if (pairs[j].count < pairs[j + 1].count) {
                ConfusionPair temp = pairs[j];
                pairs[j] = pairs[j + 1];
                pairs[j + 1] = temp;
            }
        }
    }

    printf("Actual -> Predicted (Count)\n");
    for (int i = 0; i < 5 && i < pair_count; i++) {
        printf("   %d   ->     %d     (%d times)\n",
               pairs[i].actual, pairs[i].predicted, pairs[i].count);
    }
}

int main(void) {
    printf("=== MNIST Neural Network Test ===\n\n");

    // Load network
    printf("Loading neural network...\n");
    Network *net = load_network("weights.bin");
    if (!net) {
        fprintf(stderr, "Failed to load neural network!\n");
        fprintf(stderr, "Make sure you have trained the network first by running ./train_nn\n");
        return 1;
    }

    // Load test data
    printf("Loading test data...\n");
    int num_images, num_labels;
    float *images = read_mnist_images("../files/t10k-images-idx3-ubyte", &num_images);
    unsigned char *labels = read_mnist_labels("../files/t10k-labels-idx1-ubyte", &num_labels);

    if (!images || !labels) {
        fprintf(stderr, "Failed to load test data!\n");
        free_network(net);
        free(images);
        free(labels);
        return 1;
    }

    if (num_images != num_labels) {
        fprintf(stderr, "Mismatch: %d images vs %d labels\n", num_images, num_labels);
        free_network(net);
        free(images);
        free(labels);
        return 1;
    }

    printf("Loaded %d test samples\n\n", num_images);

    // Analyze predictions
    analyze_predictions(net, images, labels, num_images);

    // Cleanup
    free_network(net);
    free(images);
    free(labels);

    return 0;
}
