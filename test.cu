#include <cuda_runtime_api.h>
// #include <curand.h>
// #include <curand_kernel.h>
#include "rt_utils.cu"
#include <math.h>
#include <stdbool.h>
#include <stdio.h>

#define gpuErrchk(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

#define PC_VAL 0.005f // Pixel to Coordinate ratio w = -3,3 | h = -2,2
#define R_COUNT 10

__global__ void test_kernel(Object *objects, int count, UCHAR *r, UCHAR *g,
                            UCHAR *b, int w, int h) {
  int x_p = blockDim.x * blockIdx.x + threadIdx.x;
  int y_p = blockDim.y * blockIdx.y + threadIdx.y;
  if (x_p >= w || y_p >= h)
    return;
  int index = x_p + y_p * w;
  unsigned int seed = index + 10;

  double x = (x_p - w / 2.0) * PC_VAL, y = (y_p - h / 2.0) * -PC_VAL;

  Vec3 r_origin;
  r_origin.x = 0;
  r_origin.y = 4.0f;
  r_origin.z = 0;
  Vec3 r_dir;
  r_dir.x = x;
  r_dir.y = y;
  r_dir.z = 6.0f;
  divide_v(&r_dir, len_v(&r_dir));
  rotateDirection(&r_dir, 10, 0, 0);

  Vec3 r_o;
  Vec3 r_d;
  Vec3 ray_energy = {.x = 0, .y = 0, .z = 0};
  Vec3 ray_color = {.x = 1, .y = 1, .z = 1};

  Vec3 intersection, normal;
  int hit_index;
  int reflect_count;

  for (int i = 0; i < R_COUNT; i++) {
    ray_color = {.x = 1, .y = 1, .z = 1};
    reflect_count = 0;
    copy_v(&r_o, &r_origin);
    copy_v(&r_d, &r_dir);
    r_o.x += my_drand(&seed) * 0.02 - 0.01;
    r_o.y += my_drand(&seed) * 0.02 - 0.01;
    while (reflect_count < 5) {
      if (find_closest_hit(&r_o, &r_d, objects, count, &intersection, &normal,
                           &hit_index)) {
        Object o = objects[hit_index];
        reflect_count++;
        copy_v(&r_o, &intersection);

        if (hit_index == 3) {
          Vec3 ref;
          reflect(&r_d, &normal, &ref);
          random_direction_hemi(&r_d, &normal, &seed);
          double rate = 0.95;
          r_d.x = lerp(r_d.x, ref.x, rate);
          r_d.y = lerp(r_d.y, ref.y, rate);
          r_d.z = lerp(r_d.z, ref.z, rate);
        } else {
          random_direction_hemi(&r_d, &normal, &seed);
        }

        if (hit_index == 0) {
          Vec3 object_enery;
          object_enery.x = 50;
          object_enery.y = 50;
          object_enery.z = 50;

          mult_v(&object_enery, &ray_color);
          add_v(&ray_energy, &object_enery);
        }

        ray_color.x *= o.color.r / 255.0f;
        ray_color.y *= o.color.g / 255.0f;
        ray_color.z *= o.color.b / 255.0f;
      } else {
        Vec3 object_enery;
        object_enery.x = 0.1;
        object_enery.y = 0.1;
        object_enery.z = 0.1;

        mult_v(&object_enery, &ray_color);
        add_v(&ray_energy, &object_enery);

        ray_color.x *= 0.663f;
        ray_color.y *= 0.949f;
        ray_color.z *= 0.961f;

        break;
      }
    }
  }

  r[index] = ray_energy.x * 255.0 / R_COUNT;
  g[index] = ray_energy.y * 255.0 / R_COUNT;
  b[index] = ray_energy.z * 255.0 / R_COUNT;
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
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
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
  Scene *scene = sample_scene_cuda();

  pipeline(scene, setting, test_renderer);

  free_scene(scene);
}
