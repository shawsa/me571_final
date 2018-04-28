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
    //read degree
    int deg = 2;
    int pdim = (deg+1)*(deg+2)/2;
    //printf("%d\t%d\t%d\n", n, nb, l);

    double *xs_local = (double*) malloc(sizeof(double)*(n+nb));
    double *ys_local = (double*) malloc(sizeof(double)*(n+nb));
    fread(xs_local, sizeof(double), n+nb, fp);
    fread(ys_local, sizeof(double), n+nb, fp);


    int *nn_local = (int*)malloc(n * l * sizeof(int));
    fread(nn_local, sizeof(int), n*l, fp);

    fclose(fp);
    
    

    for(int i=0; i<n+nb; i++){
        printf("%f\t%f\n", xs_local[i], ys_local[i]);
    }

    //printf("%d\n", nn[0][0]);

    for(int r=0; r<n; r++){
        for(int c=0; c<l; c++){
            printf("%d\t", nn_local[r*l+c]);
        }
        printf("\n");
    }


    int *nn;
    double *xs, *ys, *weights, *full_mat1_root, *RHS1_root;
    cudaMalloc((void**)&xs, (n+nb)*sizeof(double));
    cudaMalloc((void**)&ys, (n+nb)*sizeof(double));
    cudaMalloc((void**)&nn, (n*l)*sizeof(int));
    cudaMalloc((void**)&weights, n*(l+pdim)*sizeof(double));
    cudaMalloc((void**)&full_mat1_root, n*(l+pdim)*(l+pdim) * sizeof(double));
    cudaMalloc((void**)&RHS1_root, n*(l+pdim) * sizeof(double));

    cudaMemcpy(xs, xs_local, (n+nb)*sizeof(double), cudaMemcpyHostToDevice); 
    cudaMemcpy(ys, ys_local, (n+nb)*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(nn, nn_local, (n*l)*sizeof(int), cudaMemcpyHostToDevice); 


    genDMatrix<<<n,1>>>(xs, ys, nn, weights, full_mat1_root, RHS1_root, l, deg);

    double *w_local = (double*) malloc(n*(l+pdim)*sizeof(double));
    cudaMemcpy(w_local, weights, n*(l+pdim)*sizeof(double), cudaMemcpyDeviceToHost);

    for(int i=0; i < n; i++){
        for(int j=0; j<l; j++){
            printf("%f\t", w_local[i*l + j]);
        }
        printf("\n");
    }

   
    free(xs_local);
    free(ys_local);
    free(nn_local);
    free(w_local);

    cudaFree(xs);
    cudaFree(ys);
    cudaFree(nn);
    cudaFree(weights);
    cudaFree(full_mat1_root);
    cudaFree(RHS1_root);

    return 0;

}
