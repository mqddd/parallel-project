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

__device__ void pixel_ray(double x, double y, Vec3 *origin, Vec3 *direction) {
  origin->x = 0;
  origin->y = 4.0f;
  origin->z = 0;

  direction->x = x;
  direction->y = y;
  direction->z = 6.0f;
  divide_v(direction, len_v(direction));
  rotateDirection(direction, 10, 0, 0);
}

__device__ void trace_ray(Vec3 *origin, Vec3 *direction, int ray_count,
                          Object *objects, int object_count, Vec3 *ray_energy,
                          unsigned *seed) {
  Vec3 ray_color = {.x = 1, .y = 1, .z = 1};

  Vec3 intersection, normal;
  int hit_index;
  int reflect_count;
  Vec3 r_o;
  Vec3 r_d;

  for (int i = 0; i < ray_count; i++) {
    ray_color = {.x = 1, .y = 1, .z = 1};
    reflect_count = 0;
    copy_v(&r_o, origin);
    copy_v(&r_d, direction);
    r_o.x += my_drand(seed) * 0.05 - 0.025;
    r_o.y += my_drand(seed) * 0.05 - 0.025;
    while (reflect_count < 5) {
      if (find_closest_hit(&r_o, &r_d, objects, object_count, &intersection,
                           &normal, &hit_index)) {
        reflect_count++;
        float o_r = objects[hit_index].color.r;
        float o_g = objects[hit_index].color.g;
        float o_b = objects[hit_index].color.b;

        float reflect_ray = hit_index == 3 ? 1 : 0.1;
        float glow_ray = hit_index == 0 ? 2 : hit_index == 8 ? 50 : 0.0;

        copy_v(&r_o, &intersection);
        reflect(&r_d, &normal, &r_d);
        random_direction_hemi_and_lerp(&r_d, &normal, seed, 1.0 - reflect_ray);
        normalize_v(&r_d);

        Vec3 emitted_light;
        emitted_light.x = glow_ray;
        emitted_light.y = glow_ray;
        emitted_light.z = glow_ray;

        mult_v(&emitted_light, &ray_color);
        add_v(ray_energy, &emitted_light);

        ray_color.x *= o_r / 255.0f;
        ray_color.y *= o_g / 255.0f;
        ray_color.z *= o_b / 255.0f;
      } else {
        float sky_emitted_light_strength = 0.2;
        Vec3 sky_color;
        sky_color.x = 0.863f;
        sky_color.y = 0.949f;
        sky_color.z = 0.961f;
        Vec3 sky_emitted_light;
        sky_emitted_light.x = sky_emitted_light_strength;
        sky_emitted_light.y = sky_emitted_light_strength;
        sky_emitted_light.z = sky_emitted_light_strength;

        mult_v(&sky_emitted_light, &sky_color);

        mult_v(&sky_emitted_light, &ray_color);
        add_v(ray_energy, &sky_emitted_light);

        break;
      }
    }
  }

  ray_energy->x /= ray_count;
  ray_energy->y /= ray_count;
  ray_energy->z /= ray_count;
}

#define PC_VAL 0.005f // Pixel to Coordinate ratio w = -3,3 | h = -2,2
#define R_COUNT 10

__global__ void test_kernel(Object *objects, int count, UCHAR *r, UCHAR *g,
                            UCHAR *b, int w, int h, int rays) {
  int x_p = (blockDim.x * blockIdx.x + threadIdx.x) / rays;
  int y_p = blockDim.y * blockIdx.y + threadIdx.y;

  if (x_p >= w || y_p >= h)
    return;
  int index = blockDim.x * blockIdx.x + threadIdx.x + y_p * (w * rays);
  unsigned int seed = index + 10;

  double x = (x_p - w / 2.0) * PC_VAL, y = (y_p - h / 2.0) * -PC_VAL;

  Vec3 r_origin;
  Vec3 r_dir;
  pixel_ray(x, y, &r_origin, &r_dir);

  Vec3 ray_energy = {.x = 0, .y = 0, .z = 0};
  trace_ray(&r_origin, &r_dir, R_COUNT, objects, count, &ray_energy, &seed);

  r[index] = (ray_energy.x > 1 ? 1 : ray_energy.x) * 255.0;
  g[index] = (ray_energy.y > 1 ? 1 : ray_energy.y) * 255.0;
  b[index] = (ray_energy.z > 1 ? 1 : ray_energy.z) * 255.0;
}

__global__ void average_kernel(UCHAR *r, UCHAR *g, UCHAR *b, UCHAR *r_out,
                               UCHAR *g_out, UCHAR *b_out, int w, int h,
                               int rays) {
  int x_p = blockDim.x * blockIdx.x + threadIdx.x;
  int y_p = blockDim.y * blockIdx.y + threadIdx.y;
  if (x_p >= w || y_p >= h)
    return;
  int index = x_p + y_p * w;

  int rp = 0;
  int gp = 0;
  int bp = 0;
  for (int i = 0; i < rays; i++) {
    int in_index = (x_p * rays) + i + y_p * (w * rays);
    rp += r[in_index];
    gp += g[in_index];
    bp += b[in_index];
  }

  r_out[index] = rp / rays;
  g_out[index] = gp / rays;
  b_out[index] = bp / rays;
}

void test_renderer(Scene *scene, Frame *frame, PipelineSetting setting) {
  int w = frame->width;
  int h = frame->height;
  int rays = 10;
  UCHAR *r;
  cudaMalloc(&r, sizeof(UCHAR) * w * h * rays);
  UCHAR *g;
  cudaMalloc(&g, sizeof(UCHAR) * w * h * rays);
  UCHAR *b;
  cudaMalloc(&b, sizeof(UCHAR) * w * h * rays);

  UCHAR *r_out;
  cudaMalloc(&r_out, sizeof(UCHAR) * w * h);
  UCHAR *g_out;
  cudaMalloc(&g_out, sizeof(UCHAR) * w * h);
  UCHAR *b_out;
  cudaMalloc(&b_out, sizeof(UCHAR) * w * h);

  Object *d_objects;
  cudaMalloc(&d_objects, sizeof(Object) * scene->count);
  cudaMemcpy(d_objects, scene->objects, sizeof(Object) * scene->count,
             cudaMemcpyHostToDevice);

  int block_size = 16;
  dim3 thd = dim3(block_size, block_size);
  dim3 bld = dim3((w * rays - 1) / block_size + 1, (h - 1) / block_size + 1);
  printf("%d, %d\n", w, h);
  printf("%d, %d\n", bld.x, bld.y);

  float time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  test_kernel<<<bld, thd>>>(d_objects, scene->count, r, g, b, w, h, rays);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  printf("GPU kernel took %.4f ms \n\n", time);

  // ----
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  average_kernel<<<bld, thd>>>(r, g, b, r_out, g_out, b_out, w, h, rays);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  printf("Average kernel took %.4f ms \n\n", time);

  cudaMemcpy(frame->r, r_out, w * h, cudaMemcpyDeviceToHost);
  cudaMemcpy(frame->g, g_out, w * h, cudaMemcpyDeviceToHost);
  cudaMemcpy(frame->b, b_out, w * h, cudaMemcpyDeviceToHost);
  cudaFree(r);
  cudaFree(g);
  cudaFree(b);
}

int main() {
  PipelineSetting setting = {.width = 400,
                             .height = 200,
                             .debug = 1,
                             .save = 1,
                             .out_file = (char *)"test_cu.bmp"};
  Scene *scene = sample_scene_cuda();

  pipeline(scene, setting, test_renderer);

  free_scene(scene);
}
