#include <stdio.h>
#include <stdlib.h>
#include "kernels.h"

int main(int argc, char** argv){

    FILE *fp;
    fp = fopen("point_sets/test.dat", "r");
    int n, nb, l;
    
    fread(&n, sizeof(int), 1, fp);
    fread(&nb, sizeof(int), 1, fp);
    fread(&l, sizeof(int), 1, fp);
    //printf("%d\t%d\t%d\n", n, nb, l);

    double *xs_local = (double*) malloc(sizeof(double)*(n+nb));
    double *ys_local = (double*) malloc(sizeof(double)*(n+nb));
    fread(xs_local, sizeof(double), n+nb, fp);
    fread(ys_local, sizeof(double), n+nb, fp);


    int **nn = (int**)malloc(n*sizeof *nn + (n * (l * sizeof **nn)));
    int *const nn_data = (int*)nn + n;
    for(int i=0; i<n; i++){
        nn[i] = nn_data + i*l;
    }
    fread(nn_data, sizeof(int), n*l, fp);

    fclose(fp);
    
    
/*
    for(int i=0; i<n+nb; i++){
        printf("%f\t%f\n", xs_local[i], ys[i]);
    }

    //printf("%d\n", nn[0][0]);

    for(int r=0; r<n; r++){
        for(int c=0; c<l; c++){
            printf("%d\t", nn[r][c]);
        }
        printf("\n");
    }
*/

    double *xs, *ys, *b;
    cudaMalloc((void**)&xs, (n+nb)*sizeof(double));
    cudaMalloc((void**)&ys, (n+nb)*sizeof(double));
    cudaMalloc((void**)&b, n*sizeof(double));
;
    cudaMemcpy(xs, xs_local, (n+nb)*sizeof(double), cudaMemcpyHostToDevice); 
    cudaMemcpy(ys, ys_local, (n+nb)*sizeof(double), cudaMemcpyHostToDevice); 


    testKernel<<<n,1>>>(xs, b);

    double *b_local = (double*) malloc(sizeof(double)*n);
    cudaMemcpy(b_local, b, n*sizeof(double), cudaMemcpyDeviceToHost);

    for(int i=0; i < n; i++){
        printf("%f\t%f\n", b_local[i], xs_local[i]);
    }

   
    free(xs_local);
    free(ys_local);
    free(nn);

    cudaFree(xs);
    cudaFree(ys);
    cudaFree(b);
    free(b_local);


}
