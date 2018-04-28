#ifndef SYSTEM_TEST
#define SYSTEM_TEST
#include <math.h>

double dist(double *x1, double *y1, double *x2, double *y2){
	return sqrt(pow(*x1-*x2,2)+pow(*y1-*y2,2));
}

/* r^3 */
double rbf(double *x1, double *y1, double *x2, double *y2){
	return pow(dist(x1, y1, x2, y2),3);
}

/* 6*r */
double rbfd2(double *x1, double *y1, double *x2, double *y2){
	return 6*dist(x1, y1, x2, y2);
}

/* n choose k */
int choose(int n)
{
	n = n + 2;
    int ans=1;
    int k = 2;
    k=k>n-k?n-k:k;
    int j=1;
    for(;j<=k;j++,n--)
    {
        if(n%j==0)
        {
            ans*=n/j;
        }else
        if(ans%j==0)
        {
            ans=ans/j*n;
        }else
        {
            ans=(ans*n)/j;
        }
    }
    return ans;
}

/* Gaussian Elimination */
void gauss_elim(double *A, double *b, double *x, int n){

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


#endif
