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

#define cudaCheckForErrorAndSync()                                             \
  gpuErrchk(cudaPeekAtLastError());                                            \
  gpuErrchk(cudaDeviceSynchronize());

#define cudaStartTimer(start, stop)                                            \
  cudaEventCreate(&start);                                                     \
  cudaEventCreate(&stop);                                                      \
  cudaEventRecord(start, 0);

#define cudaStopTimerAndRecord(start, stop, time)                              \
  cudaEventRecord(stop, 0);                                                    \
  cudaEventSynchronize(stop);                                                  \
  cudaEventElapsedTime(&time, start, stop);

#define VP_W 0.7f
#define VP_H VP_W * 9 / 16
#define DIAFRAGM 0.002f
#define FOCAL 10
#define RAY_BOUNCE_LIMIT 10

__device__ __forceinline__ void pixel_ray(float x, float y, Vec3 *origin,
                                          Vec3 *direction) {
  origin->x = 0;
  origin->y = 4.0f;
  origin->z = 0;

  direction->x = x;
  direction->y = y;
  direction->z = FOCAL;
  // divide_v(direction, len_v(direction));
  rotateDirection(direction, 7, 0, 0);
  normalize_v(direction);
}

__device__ __forceinline__ void trace_ray(Vec3 *origin, Vec3 *direction,
                                          int ray_count, const Object *objects,
                                          int object_count, Vec3 *ray_energy,
                                          unsigned *seed) {
  Vec3 ray_color = {.x = 1, .y = 1, .z = 1};

  Vec3 intersection, normal;
  int hit_index, reflect_count, prev_hit_index = -1;
  Vec3 r_o, r_d, emitted_light;

  Vec3 sky_emitted_light;
  float sky_emitted_light_strength = 0.6;

  for (int i = 0; i < ray_count; i++) {
    ray_color = {.x = 1, .y = 1, .z = 1};
    reflect_count = 0;
    copy_v(&r_o, origin);
    copy_v(&r_d, direction);
    prev_hit_index = -1;
    move_point_randomly_in_circle(&r_o, seed, DIAFRAGM / 2);
    r_d.x += my_drand(seed) * 0.001 - 0.0005;
    r_d.y += my_drand(seed) * 0.001 - 0.0005;
    normalize_v(&r_d);
    while (reflect_count < RAY_BOUNCE_LIMIT) {
      if (find_closest_hit(&r_o, &r_d, objects, object_count, prev_hit_index,
                           &intersection, &normal, &hit_index)) {
        reflect_count++;
        const Object *obj = &objects[hit_index];
        prev_hit_index = hit_index;

        copy_v(&r_o, &intersection);
        reflect(&r_d, &normal, &r_d);
        random_direction_hemi_and_lerp(&r_d, &normal, seed,
                                       1.0 - obj->material.specular_rate);
        normalize_v(&r_d);

        emitted_light.x = obj->material.emission_color.a *
                          obj->material.emission_strength * ray_color.x;
        emitted_light.y = obj->material.emission_color.b *
                          obj->material.emission_strength * ray_color.y;
        emitted_light.z = obj->material.emission_color.c *
                          obj->material.emission_strength * ray_color.z;

        add_v(ray_energy, &emitted_light);

        ray_color.x *= obj->material.color.a;
        ray_color.y *= obj->material.color.b;
        ray_color.z *= obj->material.color.c;
      } else {
        sky_emitted_light.x =
            ((r_d.y + 0.1) * 0.1) * sky_emitted_light_strength * ray_color.x;
        sky_emitted_light.y =
            ((r_d.y + 0.1) * 0.5) * sky_emitted_light_strength * ray_color.y;
        sky_emitted_light.z =
            ((r_d.y + 0.1) * 0.9) * sky_emitted_light_strength * ray_color.z;

        add_v(ray_energy, &sky_emitted_light);
        break;
      }
    }
  }

  ray_energy->x /= ray_count;
  ray_energy->y /= ray_count;
  ray_energy->z /= ray_count;
}

__global__ void test_kernel(const Object *objects, const int count,
                            unsigned short *r, unsigned short *g,
                            unsigned short *b, const int w, const int h,
                            const int rays, const int ray_coercion) {
  int x_p = (blockDim.x * blockIdx.x + threadIdx.x) / rays;
  int y_p = blockDim.y * blockIdx.y + threadIdx.y;

  if (x_p >= w || y_p >= h)
    return;
  int index = blockDim.x * blockIdx.x + threadIdx.x + y_p * (w * rays);
  unsigned int seed = index + 10;

  float x = ((x_p - w / 2.0) / w) * VP_W * FOCAL * 2,
        y = -((y_p - h / 2.0) / h) * VP_H * FOCAL * 2;

  Vec3 r_origin;
  Vec3 r_dir;
  pixel_ray(x, y, &r_origin, &r_dir);

  Vec3 ray_energy = {.x = 0, .y = 0, .z = 0};
  trace_ray(&r_origin, &r_dir, ray_coercion, objects, count, &ray_energy,
            &seed);

  r[index] = ray_energy.x * 255.0;
  g[index] = ray_energy.y * 255.0;
  b[index] = ray_energy.z * 255.0;
}

__global__ void average_kernel(const unsigned short *r, const unsigned short *g,
                               const unsigned short *b, UCHAR *r_out,
                               UCHAR *g_out, UCHAR *b_out, int w, int h,
                               int rays) {
  int x_p = blockDim.x * blockIdx.x + threadIdx.x;
  int y_p = blockDim.y * blockIdx.y + threadIdx.y;
  if (x_p >= w || y_p >= h)
    return;
  int index = x_p + y_p * w;

  float rp = 0;
  float gp = 0;
  float bp = 0;
  for (int i = 0; i < rays; i++) {
    int in_index = (x_p * rays) + i + y_p * (w * rays);
    rp += r[in_index];
    gp += g[in_index];
    bp += b[in_index];
  }

  rp = rp / rays;
  gp = gp / rays;
  bp = bp / rays;

  float max_e = max(rp, max(gp, bp));
  if (max_e > 255) {
    rp = (rp / max_e) * 255 * 1.5;
    gp = (gp / max_e) * 255 * 1.5;
    bp = (bp / max_e) * 255 * 1.5;
  }

  r_out[index] = (rp) > 255 ? 255 : (rp);
  g_out[index] = (gp) > 255 ? 255 : (gp);
  b_out[index] = (bp) > 255 ? 255 : (bp);
}

void test_renderer(Scene *scene, Frame *frame, PipelineSetting setting) {
  int w = frame->width;
  int h = frame->height;
  int rays_thread_count = setting.ray_per_pixel / setting.cuda_coercion_rate;
  int ray_coercion = setting.ray_per_pixel / rays_thread_count;
  unsigned short *r;
  cudaMalloc(&r, sizeof(unsigned short) * w * h * rays_thread_count);
  unsigned short *g;
  cudaMalloc(&g, sizeof(unsigned short) * w * h * rays_thread_count);
  unsigned short *b;
  cudaMalloc(&b, sizeof(unsigned short) * w * h * rays_thread_count);

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
  dim3 bld = dim3((w * rays_thread_count - 1) / block_size + 1,
                  (h - 1) / block_size + 1);

  printf("Width: %d, Height: %d, Rays per pixel: %d\n", w, h,
         setting.ray_per_pixel);
  printf("Grid width: %d, Grid height: %d, Total thread count: %d\n", bld.x,
         bld.y, w * h * rays_thread_count);
  printf("Total ray per pixel: %d, Thread for each ray: %d, Rays per thread: "
         "%d\n",
         rays_thread_count * ray_coercion, rays_thread_count, ray_coercion);

  float time;
  cudaEvent_t start, stop;
  cudaStartTimer(start, stop);

  test_kernel<<<bld, thd>>>(d_objects, scene->count, r, g, b, w, h,
                            rays_thread_count, ray_coercion);
  cudaCheckForErrorAndSync();
  cudaStopTimerAndRecord(start, stop, time);
  printf("GPU kernel took %.4f ms \n\n", time);

  // ----
  cudaStartTimer(start, stop);

  average_kernel<<<bld, thd>>>(r, g, b, r_out, g_out, b_out, w, h,
                               rays_thread_count);
  cudaCheckForErrorAndSync();
  cudaStopTimerAndRecord(start, stop, time);
  printf("Average kernel took %.4f ms \n\n", time);

  cudaMemcpy(frame->r, r_out, w * h, cudaMemcpyDeviceToHost);
  cudaMemcpy(frame->g, g_out, w * h, cudaMemcpyDeviceToHost);
  cudaMemcpy(frame->b, b_out, w * h, cudaMemcpyDeviceToHost);
  cudaFree(r);
  cudaFree(g);
  cudaFree(b);
}

int main(int argc, char *argv[]) {
  int ray_count = 10;
  int width = 1200;
  int coersion_rate = 3;
  if (argc == 4) {
    width = atoi(argv[1]);
    ray_count = atoi(argv[2]);
    coersion_rate = atoi(argv[3]);
    if (width <= 0) {
      printf("Please provide a valid positive integer for width.\n");
      return 1;
    }
    if (ray_count <= 0) {
      printf("Please provide a valid positive integer for ray_per_pixel.\n");
      return 1;
    }
    if (coersion_rate <= 0) {
      printf("Please provide a valid positive integer for coersion_rate.\n");
      return 1;
    }
  }

  int height = width * 9 / 16;
  PipelineSetting setting = {.width = width,
                             .height = height,
                             .ray_per_pixel = ray_count,
                             .cuda_coercion_rate = coersion_rate,
                             .debug = 1,
                             .save = 1,
                             .out_file = (char *)"test_cu.bmp"};
  Scene *scene = sample_scene_cuda();

  pipeline(scene, setting, test_renderer);

  free_scene(scene);
}
