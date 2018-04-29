CC = nvcc

genmat: genmat.cu kernels.cu demo_util.c
	nvcc -o bin/genmat genmat.cu kernels.cu demo_util.c -lm
