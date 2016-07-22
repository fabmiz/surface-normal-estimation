#ifndef CM_H
#define CM_H

//On Rivanna
//module load gsl
//gcc -O3 -o cov_mat cov_matrix.c -lgsl -lgslcblas -lm

typedef struct {
    float x;
    float y;
    float z;
} point_rgbd;

typedef struct {
    int width;
    int height;
    point_rgbd *const *pc;
} point_cloud;

void initialize_pc(const int m, const int n, point_rgbd *const *pc);
void calc_cov_mat(int m,int n, float ***intImgs, float** dt_map);
float convert2gray(point_rgbd point);
void die(const char *error);
float area(float ** intImg,int x,int y);
float **generate_integral_image (point_rgbd *const *pci,int m, int n, int code);
float dc_smoothing_area(point_rgbd p, point_rgbd p1, point_rgbd p2, float scale_factor);
float dd_smoothing_area(point_rgbd point, float beta);
float **allocate_float_array(const int m, const int n);
point_rgbd **allocate_pc(const int width, const int height);
float **dtBin(float **dci, int m, int n);
void dt2D(float **dci, int m, int n);
float *dt1D(float *f, int n);
float **smothing_windows_map(point_rgbd *const *pc, int m, int n,float ** dt_map, int beta);
float avg(float ** intImg,int m,int n, float radius);
int max(int a, int b);
const double alpha = 0.0028;
point_cloud cloud;
float window_average(float ** intImg,int m,int n,float r);
void compute_evec(float cov_mat[3][3]);

#endif
