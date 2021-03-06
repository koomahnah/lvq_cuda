#include<cstdio>
#include <cassert>

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))

#define THREADS_PER_BLOCK 32
#define ATTRACT_STEP 0.9

extern "C" {
#define weight_index(neuron, i) ((neuron) * (input_dim) + (i))
__global__
void attract(int input_dim, double *neuron_weight, int neuron_index,
        int *text_array, int text_index, double step)
{
    int thid = blockIdx.x * blockDim.x + threadIdx.x;

    if (thid >= input_dim)
        return;

    double delta = step *
        (text_array[text_index * input_dim + thid] - neuron_weight[neuron_index * input_dim + thid]);
    neuron_weight[neuron_index * input_dim + thid] += delta;
}

__global__
void distance(int input_dim, double *neuron_weight, int *text_array, int text_index,
              double *neuron_dist)
{
    __shared__ double tab[THREADS_PER_BLOCK];
    int offset;
    int lx = threadIdx.x;
    double *this_neuron = neuron_weight + (blockIdx.x * input_dim);
    int *this_text = text_array + (text_index * input_dim);

    tab[lx] = 0;
    for (int i = 0; i < 8; i++) {
        int id = lx + i * THREADS_PER_BLOCK;
        if (id < input_dim) {
//            if (i > 0)
//                printf("reducing\n");
//            printf("neuron%d: at index%d, this_neuron[%d]=%f, this_text[%d]=%d\n",
//                    blockIdx.x, id, id, this_neuron[id], id, this_text[id]);
            double val = this_neuron[id] - this_text[id];
            tab[lx] += val * val;
        }
    }
    __syncthreads();
    for (offset = 1; offset < THREADS_PER_BLOCK; offset <<= 1) {
        double tmp = tab[lx];
        if (lx+offset < THREADS_PER_BLOCK)
            tmp = tmp + tab[lx+offset];
        __syncthreads();
        tab[lx] = tmp;
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        neuron_dist[blockIdx.x] = tab[0];
//        printf("distance of neuron%d is %f\n", blockIdx.x, tab[0]);
    }
}
__global__
void init(int input_dim, int output_dim, int neuron_count,
          double *neuron_weight)
{
    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    int i;
    double val = (thid == 0) ? 0.3 : 0.5;
    
    for (i = 0; i < input_dim; i++)
        neuron_weight[weight_index(thid, i)] = val;

}
}

