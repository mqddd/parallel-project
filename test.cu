extern "C" {
#include "pipeline.h"
}
#include <cuda_runtime_api.h>
#include <stdio.h>
// #include "vec3.h"

typedef struct {
  double x;
  double y;
  double z;
} Vec3;

__device__ void add_v(Vec3 *vec, Vec3 *target) {
  vec->x += target->x;
  vec->y += target->y;
  vec->z += target->z;
}
__device__ void add_v(Vec3 *vec, int v) {
  vec->x += v;
  vec->y += v;
  vec->z += v;
}
__device__ void sub_v(Vec3 *vec, Vec3 *target) {
  vec->x -= target->x;
  vec->y -= target->y;
  vec->z -= target->z;
}
__device__ void sub_v(Vec3 *vec, int v) {
  vec->x -= v;
  vec->y -= v;
  vec->z -= v;
}
__device__ void mult_v(Vec3 *vec, Vec3 *target) {
  vec->x *= target->x;
  vec->y *= target->y;
  vec->z *= target->z;
}
__device__ void mult_v(Vec3 *vec, int v) {
  vec->x *= v;
  vec->y *= v;
  vec->z *= v;
}
__device__ void divide_v(Vec3 *vec, int v) {
  vec->x /= v;
  vec->y /= v;
  vec->z /= v;
}
__device__ double dot_v(Vec3 *vec, Vec3 *target) {
  vec->x += target->x;
  vec->y += target->y;
  vec->z += target->z;
  return vec->x * target->x + vec->y * target->y + vec->z * target->z;
}
__device__ void cross_v(Vec3 *vec, Vec3 *target) {
  double x = vec->x;
  double y = vec->y;
  double z = vec->z;
  vec->x = y * target->z - z * target->y;
  vec->y = -x * target->z - z * target->x;
  vec->z = x * target->y - y * target->x;
}
__device__ void copy_v(Vec3 *vec, Vec3 *target) {
  vec->x = target->x;
  vec->y = target->y;
  vec->z = target->z;
}

__global__ void test_kernel(Object *objects, int count, UCHAR *r, UCHAR *g,
                            UCHAR *b, int w, int h) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  if (x >= w || y >= h)
    return;
  int index = x + y * w;
  Vec3 ray;
  ray.x = x;
  ray.y = y;
  ray.z = 0;

  Vec3 a = {.x = 255.0 / w, .y = 255.0 / h, .z = 1};
  mult_v(&ray, &a);

  r[index] = ray.x;
  g[index] = ray.y;
  b[index] = ray.z;
  // for (int i = 0; i < count; i++) {
  //   Object o = objects[i];
  //   int i_x = x - o.x;
  //   int i_y = y - o.y;
  //   if (i_x * i_x + i_y * i_y <= o.radius * o.radius) {
  //     r[index] = o.color.r;
  //     g[index] = o.color.g;
  //     b[index] = o.color.b;
  //   }
  // }
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
  printf("%d, %d\n", w, h);
  printf("%d, %d\n", bld.x, bld.y);

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
