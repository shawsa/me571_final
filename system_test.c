#include <stdio.h>
#include <stdlib.h>
#include "system_test.h"
 #include <math.h>

int main(int argc, char** argv){

    FILE *fp;
    fp = fopen("test.dat", "r");
    int n, nb, stencil_max;
    
    fread(&n, sizeof(int), 1, fp); // number of interior nodes
    fread(&nb, sizeof(int), 1, fp); // number of boundary nodes
    fread(&stencil_max, sizeof(int), 1, fp);
    printf("%d\t%d\t%d\n", n, nb, stencil_max);

    int l = stencil_max;

    double *xs = (double*) malloc(sizeof(double)*(n+nb));
    double *ys = (double*) malloc(sizeof(double)*(n+nb));
    fread(xs, sizeof(double), n+nb, fp);
    fread(ys, sizeof(double), n+nb, fp);


    int **nn = malloc(n*sizeof *nn + (n * (l * sizeof **nn))); // nearest neighbors
    int *const nn_data = nn + n;
    for(int i=0; i<n; i++){
        nn[i] = nn_data + i*l;
    }
    fread(nn_data, sizeof(int), n*l, fp);
    
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

    // Build Systems with matrices 
    //for(int k = 0; k < n; k++){
    int k = 0;
    int i, j;
    int deg = 0;
    int pdim = choose(deg);
    double full_mat1[l+pdim][l+pdim];
    double RHS1[l+pdim];

    // Make matrix 0
    for(i = 0; i < l + pdim; i++){
    	for(j = 0; j < l + pdim; j++){
    		full_mat1[i][j] = 0.0;
    	}
    }

    // Build A and O matrices
    for(i = 0; i < l + pdim; i++){
    	for(j = 0; j < l + pdim; j++){
    		if(i < l && j < l){
    			full_mat1[i][j] = rbf(&xs[nn[k][i]],&ys[nn[k][i]],&xs[nn[k][j]],&ys[nn[k][j]]);
    		}
    		else if(i >= l && j>= l){
    			full_mat1[i][j] = 0.0;
    		}
    	}
    }

    // Build P matrix
    int d = deg;
    int xp = 0;
    int yp = d;
    	for(j = l+pdim - 1; j >= l; j--){
    		for(i = 0; i < l; i++){
    			full_mat1[i][j] = pow(xs[nn[k][i]],xp)*pow(ys[nn[k][i]],yp);
    		}
    		if(yp - 1 < 0){
    			--d;
    			yp = d;
    			xp = 0;
    		}
    		else{
    			xp++;
    			yp--;    		
    		}
    	}


    // Build P transpose matrix
    d = deg;
    xp = 0;
    yp = d;
    	for(i = l+pdim - 1; i >= l; i--){
    		for(j = 0; j < l; j++){
    			full_mat1[i][j] = pow(xs[nn[k][j]],xp)*pow(ys[nn[k][j]],yp);
    		}
    		if(yp - 1 < 0){
    			--d;
    			yp = d;
    			xp = 0;
    		}
    		else{
    			xp++;
    			yp--;    		
    		}
    	}

    // RHS vector
    for(i = 0; i < l + pdim; i++){
    	if(i < l){
    		RHS1[i] = rbfd2(&xs[nn[k][0]],&ys[nn[k][0]],&xs[nn[k][i]],&ys[nn[k][i]]);
    	}
    	else{
    		RHS1[i] = 0.0;
    	}
    }

    double *w;

    //w = gauss_elim(&full_mat,RHS,l+pdim);
    for(int c=0; c<l+pdim; c++){
        //printf("%f\n", w[c]);
    }

   //Print full matrix and RHS
    for(int r=0; r<l+pdim; r++){
        for(int c=0; c<l+pdim; c++){
            printf("%f ", full_mat1[r][c]);
        }
        printf("\n");
    }
    printf("\n");
    for(int c=0; c<l+pdim; c++){
        printf("%f\n", RHS1[c]);
    }
//}



    fclose(fp);
    free(xs);
    free(ys);
    free(nn);

    return 0;
}
