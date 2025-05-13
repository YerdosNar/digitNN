#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "kmeans.h"

struct KMeans {
    int k;
    int num_points;
    int dimension;
    Point* points;
    Point* centroids;
};

static float distance(const Point *p1, const Point *p2, int dimension) {
    float sum = 0.0f;
    for(int i = 0; i < dimension; i++) {
        float diff = p1->coordinates[i] - p2->coordinates[i];
        sum += diff * diff;
    }
    return sqrtf(sum);
}

static void assign_clusters(KMeans *kmeans) {
    for(int i = 0; i < kmeans->num_points; i++) {
        Point *point = &kmeans->points[i];
        float min_dis = INFINITY;
        int cluster = -1;
        for(int j = 0; j < kmeans->k; j++) {
            float dist = distance(point, &kmeans->centroids[j], kmeans->dimension);
            if(dist < min_dis) {
                min_dis = dist;
                cluster = j;
            }
        }
        point->cluster = cluster;
    }
}

static int update_centroids(KMeans *kmeans) {
    int changed = 0;
    int dimension = kmeans->dimension;
    float **sums = malloc(kmeans->k * sizeof(float*));
    int *counts = calloc(kmeans->k, sizeof(int));

    if(!sums || !counts) { // just checking if it did allocate
        free(sums);
        free(counts);
        return -1;
    }

    for(int i = 0; i < kmeans->k; i++) {
        sums[i] = calloc(dimension, sizeof(float));
        if(!sums[i]) {
            for(int j = 0; j < i; j++) {
                free(sums[j]);
            }
            free(sums);
            free(counts);
            return -1;
        }
    }

    for(int i = 0; i < kmeans->num_points; i++) {
        int cluster = kmeans->points[i].cluster;
        if(cluster < 0 || cluster >= kmeans->k) {
            continue;
        }
        counts[cluster]++;
        for(int d = 0; d < dimension; d++) {
            sums[cluster][d] += kmeans->points[i].coordinates[d];
        }
    }

    for(int i = 0; i < kmeans->k; i++) {
        if(counts[i] == 0) {
            continue;
        }
        for(int d= 0; d < dimension; d++) {
            float new_coords = sums[i][d] / counts[i];
            if(kmeans->centroids[i].coordinates[d] != new_coords) {
                kmeans->centroids[i].coordinates[d] = new_coords;
                changed = 1;
            }
        }
    }

    for(int i = 0; i < kmeans->k; i++) {
        free(sums[i]);
    }
    free(sums);
    free(counts);
    return changed;
}

KMeans* kmeans_create(int k, Point *points, int num_points, int dimension) {
    if(k <= 0 || num_points < k || !points || dimension <= 0) {
        return NULL; // just a regular check
    }

    KMeans *kmeans = malloc(sizeof(KMeans));
    if(!kmeans) return NULL;

    kmeans->k = k;
    kmeans->num_points = num_points;
    kmeans->dimension = dimension;
    kmeans->points = malloc(num_points * sizeof(Point));

    if(!kmeans->points) {
        free(kmeans->points);
        return NULL;
    }

    for(int i = 0; i < num_points; i++) {
        kmeans->points[i].coordinates = malloc(dimension * sizeof(float));
        if(!kmeans->points[i].coordinates) {
            for(int j = 0; j < i; j++) {
                free(kmeans->points[j].coordinates);
            }
            free(kmeans->points);
            free(kmeans);
            return NULL;
        }
        memcpy(kmeans->points[i].coordinates, points[i].coordinates, dimension * sizeof(float));
        kmeans->points[i].cluster = -1;
    }

    kmeans->centroids = malloc(k * sizeof(Point));
    if(!kmeans->centroids) {
        for(int i = 0; i < num_points; i++) {
            free(kmeans->points[i].coordinates);
        }
        free(kmeans->points);
        free(kmeans);
        return NULL;
    }

    int *indices = malloc(num_points * sizeof(int));
    if(!indices) {
        for(int i = 0; i < num_points; i++) {
            free(kmeans->points[i].coordinates);
        }
        free(kmeans->points);
        free(kmeans);
        return NULL;
    }

    for(int i = 0; i < num_points; i++) {
        indices[i] = i;
    }
    for(int i= num_points - 1; i > 0; i--) {
        int j = rand() % (i+1);
        int temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }

    for(int i = 0; i < k; i++) {
        int idx = indices[i];
        kmeans->centroids[i].coordinates = malloc(dimension * sizeof(float));
        if(!kmeans->centroids[i].coordinates) {
            for(int j = 0; j < i; j++) {
                free(kmeans->centroids[j].coordinates);
            }
            free(indices);
            for(int j = 0; j < num_points; j++) {
                free(kmeans->points[j].coordinates);
            }
            free(kmeans->points);
            free(kmeans->centroids);
            free(kmeans);
            return NULL;
        }
        memcpy(kmeans->centroids[i].coordinates, kmeans->points[idx].coordinates, dimension * sizeof(float));
    }

    free(indices);
    return kmeans;
}

void kmeans_run(KMeans *kmeans, int max_iter) {
    if(!kmeans) return;
    for(int iter = 0; iter < max_iter; iter++) {
        assign_clusters(kmeans);
        if(!update_centroids(kmeans)) {
            break;
        }
    }
}

const Point* kmeans_get_centroids(const KMeans *kmeans) {
    return kmeans ? kmeans->centroids : NULL;
}

const Point* kmeans_get_points(const KMeans *kmeans) {
    return kmeans ? kmeans->points : NULL;
}

void kmeans_free(KMeans *kmeans) {
    if(!kmeans) return;
    for(int i= 0; i < kmeans->num_points; i++) {
        free(kmeans->points[i].coordinates);
    }
    free(kmeans->points);
    for(int i= 0; i < kmeans->k; i++) {
        free(kmeans->centroids[i].coordinates);
    }
    free(kmeans->centroids);
    free(kmeans);
}

