//
// Created by Evaristo Koyama on 5/11/16.
//

#ifndef SEGMENTATION_SDC_EXAMPLE_H
#define SEGMENTATION_SDC_EXAMPLE_H

typedef struct {
	float x;
	float y;
    float z;
} point_cloud;


point_cloud **allocate_raster(const int n, const int m);

void initialize_raster(const int n, const int m, point_cloud *const *pc);

void compute_sdc(const int n, const int m, point_cloud *const *pc);

float **allocate_float_array(const int n, const int m);

#endif //SEGMENTATION_SDC_EXAMPLE_H
