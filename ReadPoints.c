#include <stdio.h>
#include <stdlib.h>


void main(int argc, char** argv){

    FILE *fp;
    fp = fopen("point_sets/test.dat", "r");
    int n, nb, l;
    
    fread(&n, sizeof(int), 1, fp);
    fread(&nb, sizeof(int), 1, fp);
    fread(&l, sizeof(int), 1, fp);
    printf("%d\t%d\t%d\n", n, nb, l);

    double *xs = (double*) malloc(sizeof(double)*(n+nb));
    double *ys = (double*) malloc(sizeof(double)*(n+nb));
    fread(xs, sizeof(double), n+nb, fp);
    fread(ys, sizeof(double), n+nb, fp);


    int **nn = malloc(n*sizeof *nn + (n * (l * sizeof **nn)));
    int * const nn_data = nn + n;
    for(int i=0; i<n; i++){
        nn[i] = nn_data + i*l;
    }
    fread(nn_data, sizeof(int), n*l, fp);
    //int nn[n][l] = &nnread;
    
    for(int i=0; i<n+nb; i++){
        printf("%f\t%f\n", xs[i], ys[i]);
    }

    //printf("%d\n", nn[0][0]);

    for(int r=0; r<n; r++){
        for(int c=0; c<l; c++){
            printf("%d\t", nn[r][c]);
        }
        printf("\n");
    }


    fclose(fp);
    free(xs);
    free(ys);
    free(nn);
}
