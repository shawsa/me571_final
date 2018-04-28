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

    double *xs = (double*) malloc(sizeof(double)*(n+nb));
    double *ys = (double*) malloc(sizeof(double)*(n+nb));
    fread(xs, sizeof(double), n+nb, fp);
    fread(ys, sizeof(double), n+nb, fp);


    int **nn = malloc(n*sizeof *nn + (n * (stencil_max * sizeof **nn))); // nearest neighbors
    int *const nn_data = nn + n;
    for(int i=0; i<n; i++){
        nn[i] = nn_data + i*stencil_max;
    }

    fread(nn_data, sizeof(int), n*stencil_max, fp);
    for(int i=0; i<n+nb; i++){
        printf("%f\t%f\n", xs[i], ys[i]);
    }

    //printf("%d\n", nn[0][0]);

    for(int r=0; r<n; r++){
        for(int c=0; c<stencil_max; c++){
            printf("%d\t", nn[r][c]);
        }
        printf("\n");
    }
    /*****************************************************************************************/
    /* Build Systems in Arrays */
    //for(int k = 0; k < n; k++){
    int i, j, k, idx;
    int deg = 1;
    int pdim = choose(deg);
    int l = stencil_max;
    int sz = l+pdim;
    double *full_mat, *RHS, *w;

    full_mat = (double*) malloc(sz*sz*sizeof(double));
    RHS      = (double*) malloc(sz*sizeof(double));
    w        = (double*) malloc(sz*sizeof(double));

    // Make matrix 0
    for(k = 0; k < sz*sz; k++){
    	full_mat[k] = 0.0;
    }

    // Build A and O matrices
    k = 0;
    for(i = 0; i < sz; i++){
    	for(j = 0; j < sz; j++){
    		idx = i*sz + j;
    		if(i < l && j < l){
    			full_mat[idx] = rbf(&xs[nn[k][i]],&ys[nn[k][i]],&xs[nn[k][j]],&ys[nn[k][j]]);
    		}
    		else if(i >= l && j>= l){
    			full_mat[idx] = 0.0;
    		}
    	}
    }

    // Build P matrix
    int d = deg;
    int xp = 0;
    int yp = d;
    	for(j = sz - 1; j >= l; j--){
    		for(i = 0; i < l; i++){
    			idx = i*sz + j;
    			full_mat[idx] = pow(xs[nn[k][i]],xp)*pow(ys[nn[k][i]],yp);
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
    	for(i = sz - 1; i >= l; i--){
    		for(j = 0; j < l; j++){
    			idx = i*sz + j;
    			full_mat[idx] = pow(xs[nn[k][j]],xp)*pow(ys[nn[k][j]],yp);
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
    for(i = 0; i < sz; i++){
    	if(i < l){
    		RHS[i] = rbfd2(&xs[nn[k][0]],&ys[nn[k][0]],&xs[nn[k][i]],&ys[nn[k][i]]);
    	}
    	else{
    		RHS[i] = 0.0;
    	}
    }

    /*//Print full matrix and RHS
    k = 0;
    for(int r=0; r < sz; r++){
        for(int c=0; c < sz; c++){
            printf("%f ", full_mat[k]);
            k++;
        }
        printf("\n");
    }
    printf("\n");
    for(int c=0; c<sz; c++){
        printf("%f\n", RHS[c]);
    }
    printf("\n");*/

    gauss_elim(full_mat,RHS,w,sz);
    
    for(int c=0; c<sz; c++){
        printf("%f\n", w[c]);
    }

//}
  

    fclose(fp);
    free(xs);
    free(ys);
    free(nn);

    return 0;
}