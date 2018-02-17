#include<cstdio>
#include <cassert>

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))

#define THREADS_PER_BLOCK 32
extern "C" {
#define weight_index(neuron, i) ((neuron) * (input_dim) + (i))
__global__
void distance(int input_dim, float *neuron_weight, int *text_array, int text_index,
              float *neuron_dist)
{
    __shared__ float tab[THREADS_PER_BLOCK];
    int offset;
    int lx = threadIdx.x;
    float *this_neuron = neuron_weight + (blockIdx.x * input_dim);
    int *this_text = text_array + (text_index * input_dim);

    tab[lx] = 0;
    for (int i = 0; i < 8; i++) {
        int id = lx + i * THREADS_PER_BLOCK;
        if (id < input_dim) {
            if (i > 0)
                printf("reducing\n");
            printf("neuron%d: at index%d, this_neuron[%d]=%f, this_text[%d]=%d\n",
                    blockIdx.x, id, id, this_neuron[id], id, this_text[id]);
            float val = this_neuron[id] - this_text[id];
            tab[lx] += val * val;
        }
    }
    __syncthreads();
    for (offset = 1; offset < THREADS_PER_BLOCK; offset <<= 1) {
        float tmp = tab[lx];
        if (lx+offset < THREADS_PER_BLOCK)
            tmp = tmp + tab[lx+offset];
        __syncthreads();
        tab[lx] = tmp;
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        neuron_dist[blockIdx.x] = tab[0];
        printf("distance of neuron%d is %f\n", blockIdx.x, tab[0]);
    }
}
__global__
void init(int input_dim, int output_dim, int neuron_count, int *neuron_class,
          float *neuron_weight, int *neuron_bias)
{
    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    int i;
    float val = (thid == 0) ? 0.3 : 0.5;
    
    neuron_class[thid] = thid * output_dim / neuron_count;
    for (i = 0; i < input_dim; i++)
        neuron_weight[weight_index(thid, i)] = val;
    neuron_bias[thid] = 0;

}
}

