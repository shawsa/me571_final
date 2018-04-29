

__global__ void genDMatrix(double *xs, double *ys, 
        int *nn, double *weights, 
        double *full_mat1_root, double *RHS1_root,
        int l_max, int l, int deg);
