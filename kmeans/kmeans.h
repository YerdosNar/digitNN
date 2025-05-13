#ifndef KMEANS_H
#define KMEANS_H

typedef struct {
    float *coordinates;
    int cluster;
} Point;

typedef struct KMeans KMeans;

KMeans* kmeans_create(int k, Point *point, int num_points, int dimension);
void kmeans_run(KMeans *kmeans, int max_iter);
const Point* kmeans_get_centroids(const KMeans *kmeans);
const Point* kmeans_get_points(const KMeans *kmeans);
void kmeans_free(KMeans *kmeans);

#endif

