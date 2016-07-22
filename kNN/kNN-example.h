//
// Created by Marcus Rosti on 5/10/16.
//

#ifndef SEGMENTATION_KNN_EXAMPLE_H
#define SEGMENTATION_KNN_EXAMPLE_H

typedef struct {
    float z;
    /*float r;
    float g;
    float b;
    float a;*/
} point_cloud;


point_cloud **allocate_raster(const int n, const int m);

void initialize_raster(const int n, const int m, point_cloud *const *pc);

void compute_knn(const int n, const int m, point_cloud *const *pc, const int k);

float **allocate_float_array(const int n, const int m);

#endif //SEGMENTATION_KNN_EXAMPLE_H
