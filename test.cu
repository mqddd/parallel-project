extern "C" {
#include "pipeline.h"
}
#include <cuda_runtime_api.h>
#include <stdio.h>
// #include "vec3.h"

__global__ void test_kernel(Object *objects, int count, UCHAR *r, UCHAR *g,
                            UCHAR *b, int w, int h) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  if (x >= w || y >= h)
    return;
  int index = x + y * w;
  for (int i = 0; i < count; i++) {
    Object o = objects[i];
    int i_x = x - o.x;
    int i_y = y - o.y;
    if (i_x * i_x + i_y * i_y <= o.radius * o.radius) {
      r[index] = o.color.r;
      g[index] = o.color.g;
      b[index] = o.color.b;
    }
  }
}

void test_renderer(Scene *scene, Frame *frame, PipelineSetting setting) {
  int w = frame->width;
  int h = frame->height;
  UCHAR *r;
  cudaMalloc(&r, sizeof(UCHAR) * w * h);
  UCHAR *g;
  cudaMalloc(&g, sizeof(UCHAR) * w * h);
  UCHAR *b;
  cudaMalloc(&b, sizeof(UCHAR) * w * h);

  Object *d_objects;
  cudaMalloc(&d_objects, sizeof(Object) * scene->count);
  cudaMemcpy(d_objects, scene->objects, sizeof(Object) * scene->count,
             cudaMemcpyHostToDevice);

  int block_size = 16;
  dim3 thd = dim3(block_size, block_size);
  dim3 bld = dim3((w - 1) / block_size + 1, (h - 1) / block_size + 1);

  float time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  test_kernel<<<bld, thd>>>(d_objects, scene->count, r, g, b, w, h);
  cudaDeviceSynchronize();
  cudaDeviceSynchronize();
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  printf("GPU kernel took %.4f ms \n\n", time);

  cudaMemcpy(frame->r, r, w * h, cudaMemcpyDeviceToHost);
  cudaMemcpy(frame->g, g, w * h, cudaMemcpyDeviceToHost);
  cudaMemcpy(frame->b, b, w * h, cudaMemcpyDeviceToHost);
  cudaFree(r);
  cudaFree(g);
  cudaFree(b);
}

int main() {
  PipelineSetting setting = {.width = 1200,
                             .height = 800,
                             .debug = 1,
                             .save = 1,
                             .out_file = (char *)"test_cu.bmp"};
  Scene *scene = sample_scene1();

  pipeline(scene, setting, test_renderer);

  free_scene(scene);
}
