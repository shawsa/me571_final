#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "kernels.h"
#include "demo_util.h"


//debugging vars
#define DBG1 1 //prints xs, ys, nn
#define DBG2 2 //prints first stencil matrix
#define DBG3 4 //prints calculated weights including PB weights


int main(int argc, char** argv){

    /*************************************************************************
    *
    * Read command line args
    *
    *************************************************************************/
    
    int err;
    char filename[100];
    //strcpy(filename, "point_sets/test.dat");
    read_string(argc, argv, (char*)"-f", filename, &err);
    if(err!=0){
        strcpy(filename, "test.dat");
    }

    //printf(filename, "\n");

    char fullPath[150];
    strcpy(fullPath, "point_sets/");
    strcat(fullPath, filename);
    printf("\nReading from ");
    printf(fullPath);
    printf("\n\n");
    

    int deg, l, debug, tpb;
    err = 0;
    read_int(argc, argv, (char*)"-d", &deg, &err);
    if(err!=0){deg = 0;}
    
    err = 0;
    read_int(argc, argv, (char*)"-l", &l, &err);
    if(err!=0){l = 3;}

    printf("deg:%d\tstencil:%d\n\n", deg, l);

    err = 0;
    read_int(argc, argv, (char*)"--debug", &debug, &err);
    if(err!=0){debug = 0;}

    err = 0;
    read_int(argc, argv, (char*)"--tpb", &tpb, &err);
    if(err!=0){tpb = 1024;}


    /*************************************************************************
    *
    * Read file
    *
    *************************************************************************/

    FILE *fp;
    //fp = fopen("point_sets/test.dat", "r");
    fp = fopen(fullPath, "r");
    int n, nb, l_max;
    
    fread(&n, sizeof(int), 1, fp);
    fread(&nb, sizeof(int), 1, fp);
    fread(&l_max, sizeof(int), 1, fp);

    //Read 2 dummy ints. This is for compatability with reading matrix files
    int fake_int;
    fread(&fake_int, sizeof(int), 1, fp);
    fread(&fake_int, sizeof(int), 1, fp);

    //calculate number of pbasis terms = deg+1 choose 2
    int pdim = (deg+1)*(deg+2)/2;

    //Print n, nb, and max stencil size
    printf("n:%d\tnb:%d\tmax stencil:%d\n\n", n, nb, l_max);

    double *xs_local = (double*) malloc(sizeof(double)*(n+nb));
    double *ys_local = (double*) malloc(sizeof(double)*(n+nb));
    fread(xs_local, sizeof(double), n+nb, fp);
    fread(ys_local, sizeof(double), n+nb, fp);


    int *nn_local = (int*)malloc(n * l_max * sizeof(int));
    fread(nn_local, sizeof(int), n*l_max, fp);

    fclose(fp);

    if(l > l_max){
        printf("Error: stencil size (%d) > max size (%d)\n\n", l, l_max);
        return 1;
    }
    
    
    
    /**************************************************************************
    *
    * Start clock
    *
    **************************************************************************/
    clock_t begin_time, end_time;
    begin_time = clock();
    
    /**************************************************************************
    *
    * Print xs, ys, and nns
    *
    **************************************************************************/
    if(debug & DBG1){
        printf("xs\t\tys\n");
        for(int i=0; i<n+nb; i++){
            printf("%f\t%f\n", xs_local[i], ys_local[i]);
        }
        printf("\nNearest Neighbors\n");
        for(int r=0; r<n; r++){
            for(int c=0; c<l_max; c++){
                printf("%d\t", nn_local[r*l_max+c]);
            }
                    printf("\n");
        }
    } 

    int *nn;
    double *xs, *ys, *weights, *full_mat1_root, *RHS1_root;
    cudaMalloc((void**)&xs, (n+nb)*sizeof(double));
    cudaMalloc((void**)&ys, (n+nb)*sizeof(double));
    cudaMalloc((void**)&nn, (n*l_max)*sizeof(int));
    cudaMalloc((void**)&weights, n*(l_max+pdim)*sizeof(double));
    cudaMalloc((void**)&full_mat1_root, n*(l_max+pdim)*(l_max+pdim) * sizeof(double));
    cudaMalloc((void**)&RHS1_root, n*(l_max+pdim) * sizeof(double));

    cudaMemcpy(xs, xs_local, (n+nb)*sizeof(double), cudaMemcpyHostToDevice); 
    cudaMemcpy(ys, ys_local, (n+nb)*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(nn, nn_local, (n*l_max)*sizeof(int), cudaMemcpyHostToDevice); 

    dim3 block(tpb);
    dim3 grid((n+tpb-1)/tpb);
    genDMatrix<<<block,grid>>>(xs, ys, nn, weights, full_mat1_root, RHS1_root, l_max, l, deg);


    /**************************************************************************
    *
    * Print first stencil system
    *
    **************************************************************************/
    if(debug & DBG2){
        double *test = (double*) malloc(sizeof(double)*(l+pdim)*(l+pdim));
        double *test2 = (double*) malloc(sizeof(double)*(l+pdim));
        cudaMemcpy(test, full_mat1_root, 
                    (l+pdim)*(l+pdim) * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(test2, RHS1_root, (l+pdim) * sizeof(double), cudaMemcpyDeviceToHost);

        printf("\nFirst stencil:\n");
        for(int i=0; i<l+pdim; i++){
            for(int j=0; j<l+pdim; j++){
                printf("%f\t", test[i*(l+pdim)+j]);
            }
            printf("\n");
        }
        for(int i=0; i<l+pdim; i++){
            printf("%f\n", test2[i]);
        }
        free(test);
        free(test2);
    }



    double *w_local = (double*) malloc(n*(l_max+pdim)*sizeof(double));
    cudaMemcpy(w_local, weights, n*(l_max+pdim)*sizeof(double), cudaMemcpyDeviceToHost);


    /*************************************************************************
    *
    * Print sparse weight matrix
    *
    **************************************************************************/
    if(debug & DBG3){
        printf("\nWeigts:\n");
        for(int i=0; i < n; i++){
            for(int j=0; j<l+pdim; j++){
                printf("%f\t", w_local[i*(l+pdim) + j]);
            }
            printf("\n");
        }
    }


    /*************************************************************************
    *
    * Stop timer
    *
    **************************************************************************/
    end_time = clock();
    printf("\n\nRuntime: %fms\n", ((double)(end_time - begin_time)) * 1000.0/CLOCKS_PER_SEC);

    /*************************************************************************
    *
    * Write to output file
    * n, nb, l_max, l, deg
    * xs, ys, nn
    * weights
    *
    **************************************************************************/

    char outPath[150];
    strcpy(outPath, "weights/");
    strcat(outPath, filename);
    strcat(outPath, ".mat");
    printf("\nWriting to ");
    printf(outPath);
    printf("\n\n");



    fp = fopen(outPath, "w");
    
    fwrite(&n, sizeof(int), 1, fp);
    fwrite(&nb, sizeof(int), 1, fp);
    fwrite(&l_max, sizeof(int), 1, fp);
    fwrite(&l, sizeof(int), 1, fp);
    fwrite(&deg, sizeof(int), 1, fp);

    fwrite(xs_local, sizeof(double), n+nb, fp);
    fwrite(ys_local, sizeof(double), n+nb, fp);
    fwrite(nn_local, sizeof(int), n*l_max, fp);

    fwrite(w_local, sizeof(double), n*(l_max+pdim), fp);

    fclose(fp);


    //Free memory
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
