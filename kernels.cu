#include <math.h>

__device__ double dist(double x1, double y1, double x2, double y2){
    return sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2));
}





__global__ void testKernel(double *xs, double *ys, double *b){
    b[blockIdx.x] = dist(xs[blockIdx.x], 1.0, ys[blockIdx.x], 1.0);
}





/* r^3 */
__device__ double rbf(double x1, double y1, double x2, double y2){
	return pow(dist(x1, y1, x2, y2),3);
}

/* 6*r */
__device__ double rbfd2(double x1, double y1, double x2, double y2){
	return 6*dist(x1, y1, x2, y2);
}


/* Gaussian Elimination */
__device__ void gauss_elim(double *A, double *b, double *x, int n){

	int i, j, k;
	int idxi, idxj, idxij, idxik, idxjk;
	double m, diff;

	// Swap first and second rows
	int r1 = 0;
    int r2 = 1;
    double mtemp, vtemp;
    int idx1;
    int idx2;
    for (i = 0; i < n; ++i)
    {
        // matrix swap
        idx1 = r1*n + i;
        idx2 = r2*n + i;
        mtemp = A[idx1];
        A[idx1] = A[idx2];
        A[idx2] = mtemp;
    }

    // RHS vector swap
    vtemp = b[1];
    b[1] = b[0];
    b[0] = vtemp;

	// Gauss-Jordan Forward Elimination to Upper triangular matrix
	for (j = 0; j < n-1; j++){
        for (i = j+1; i < n; i++){
        	idxij = i*n + j;
        	idxj = j*n + j;
            m = A[idxij]/A[idxj];
            for (k = 0; k < n; k++){
            	idxik = i*n + k;
        		idxjk = j*n + k;
                A[idxik] = A[idxik] - m*A[idxjk];
            }
            b[i] = b[i] - m*b[j];
        }
    }

    // Back substituion
    for (i = n-1; i >= 0; i--){
        diff = b[i];
        for (j = i+1; j < n; j++){
        	idxij = i*n + j;
            diff = diff - x[j]*A[idxij];
        }
        idxi = i*n + i;
        x[i] = diff/A[idxi];
	}

}

__device__ void build_stencil_matrix(double* xs, double* ys, 
                    int* nn,  double* full_mat1, double* RHS1,
                    int l_max, int l, int deg, int k){
    int pdim = (deg+1)*(deg+2)/2;
    int i, j;
    // Make matrix 0
    for(i = 0; i < l + pdim; i++){
    	for(j = 0; j < l + pdim; j++){
    		full_mat1[i*(l+pdim) + j] = 0.0;
    	}
    }

    // Build A and O matrices
    for(i = 0; i < l + pdim; i++){
    	for(j = 0; j < l + pdim; j++){
    		if(i < l && j < l){
    			full_mat1[i*(l+pdim)+j] = rbf(
                            xs[nn[k*l_max+i]], ys[nn[k*l_max+i]], 
                            xs[nn[k*l_max+j]], ys[nn[k*l_max+j]]);
    		}
    		else if(i >= l && j>= l){
    			full_mat1[i*(l+pdim) + j] = 0.0;
    		}
    	}
    }

    // Build P matrix
    int d = deg;
    int xp = 0;
    int yp = d;
    	for(j = l+pdim - 1; j >= l; j--){
    		for(i = 0; i < l; i++){
    			full_mat1[i*(l+pdim) + j] = 
                        pow(xs[nn[k*l_max+i]] - xs[nn[k*l_max+0]], xp) * 
                        pow(ys[nn[k*l_max+i]] - ys[nn[k+l_max+0]], yp);
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
    			//full_mat1[i*(l+pdim) + j] = pow(xs[nn[k*l+j]],xp)*pow(ys[nn[k*l+j]],yp);
    			full_mat1[i*(l+pdim) + j] = full_mat1[j*(l+pdim) + i];
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
    		RHS1[i] = rbfd2(
                    xs[nn[k*l_max+0]], ys[nn[k*l_max+0]],
                    xs[nn[k*l_max+i]], ys[nn[k*l_max+i]]);
    	}
        else if(i==l+3 || i==l+5){
            RHS1[i] = 2.0;
        }
    	else{
    		RHS1[i] = 0.0;
    	}
    }
}


__global__ void genDMatrix(int n, double* xs, double* ys, 
                    int* nn, double* weights_root, 
                    double* full_mat1_root, double* RHS1_root,
                    int l_max, int l, int deg){

    int my_id = blockDim.x*blockIdx.x + threadIdx.x;
    int pdim = (deg+1)*(deg+2)/2; 
   
    if(my_id <n){ 
        double* full_mat1 = &full_mat1_root[my_id * (l+pdim)*(l+pdim)];
        double* RHS1 = &RHS1_root[my_id * (l+pdim)];
        double* weights = &weights_root[my_id * (l+pdim)];

        build_stencil_matrix(xs, ys, nn, full_mat1, RHS1, l_max, l, deg, my_id); 
        gauss_elim(full_mat1, RHS1, weights, l+pdim); 
    }
}

