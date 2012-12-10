// Gregory Paton
// 322:451
// CUDA Mandelbrot

#include <stdio.h>
#include <string.h>
#include <math.h>

#define		X_RESN	800       /* x resolution */
#define		Y_RESN	800       /* y resolution */

typedef struct complextype
{
    float real, imag;
} Complex;

__global__
void work(int *id, int tb_x, int tb_y, int gr_x, int gr_y)
{
    int i, j, k;
    Complex z, c;
    float lengthsq, temp;
    const int num_threads = tb_x * tb_y;
    int work_width = X_RESN / num_threads;
    const int bid = blockIdx.x + (blockIdx.y * gr_x);
    const int tid = threadIdx.x + (threadIdx.y * tb_x) + (bid * num_threads);
    id[tid] = tid;
    // if X_RESN is not evenly divisible by num_threads
    // give remainder of work to last thread
    if (tid == num_threads - 1)
        work_width = X_RESN - tid;
    
    /*for(i = tid; i < tid + work_width; i++) {
        for(j = 0; j < Y_RESN; j++) {

            z.real = z.imag = 0.0;
            c.real = ((float) j - 400.0)/200.0;
            c.imag = ((float) i - 400.0)/200.0;
            k = 0;

            do  {                              
                temp = z.real*z.real - z.imag*z.imag + c.real;
                z.imag = 2.0*z.real*z.imag + c.imag;
                z.real = temp;
                lengthsq = z.real*z.real+z.imag*z.imag;
                k++;
            } while (lengthsq < 4.0 && k < 100);
            
            if (k == 100) 
                id[i + (j * Y_RESN)] = 1;
            else
                id[i + (j * Y_RESN)] = 0;
        }
    }*/   
}

int main (int argc, char **argv)
{  
    int tb_x = 16;
    int tb_y = 1;
    int gr_x = 1;
    int gr_y = 1;
    if (argc == 5) {
        tb_x = atoi(argv[1]);
        tb_y = atoi(argv[2]);
        gr_x = atoi(argv[3]);
        gr_y = atoi(argv[4]);    
    }
    else {
        printf("usage: %s THREAD_BLOCK_WIDTH THREAD_BLOCK_HEIGHT GRID_WIDTH GRID_HEIGHT\n", argv[0]);
        return -1;
    }
    int id[X_RESN * Y_RESN];
    int *Id;
    cudaMalloc((void**)&Id, X_RESN * Y_RESN); 

    dim3 dimBlock(tb_x, tb_y);
    dim3 dimGrid(gr_x, gr_y);

    work<<<dimGrid, dimBlock>>>(Id, tb_x, tb_y, gr_x, gr_y);
    
    cudaMemcpy(id, Id, X_RESN * Y_RESN, cudaMemcpyDeviceToHost);
    int i;
    for (i = 0; i < 30; ++i)
        printf("id: %d\n", id[i]);
    cudaFree(Id);
	/* Program Finished */
    return 0;
}

