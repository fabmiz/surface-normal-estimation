#include <stdio.h>
#include <stdlib.h>

typedef struct {
  float z;
  float r;
  float g;
  float b;
  float a;
} point_cloud;


point_cloud **allocate_raster(const int n, const int m);

void initalize_raster(const int n, const int m, point_cloud *const *pc);

int main(const int argc, const char ** argv) {
  if (argc != 4) {
    printf("Takes three arguments\n");
    printf("\t{n-dim} {m-dim} {seed}\n");
    exit(0);
  }
  const int n = atoi(argv[1]);
  const int m = atoi(argv[2]);
  const int seed = atoi(argv[3]);

  srand((unsigned) seed);

  point_cloud **pc = allocate_raster(n, m);
  initalize_raster(n, m, pc);

  //compute
  //
  //compare
  for(int i = 0; i < n; i++) {
    printf("%5i,0 %2.5f\n",i,(double) pc[i][0].z);
  }

  for(int i = 0; i < n; i ++){
    free(pc[i]);
  }
  free(pc);
  return 0;
}

void initalize_raster(const int n, const int m, point_cloud *const *pc) {
  int i, j;
  for(i = 0; i < n; i++) {
    for( j = 0; j < m; j ++) {
        pc[i][j].z = rand();
        pc[i][j].r = rand();
        pc[i][j].g = rand();
        pc[i][j].b = rand();
        pc[i][j].a = rand();

      }
  }
}

point_cloud **allocate_raster(const int n, const int m) {
  int i;
  point_cloud ** pc = (point_cloud **)malloc((unsigned) n * sizeof(point_cloud *));
  for (i=0; i<n; i++)
    pc[i] = (point_cloud *)malloc((unsigned) m * sizeof(point_cloud));
  return pc;
}

