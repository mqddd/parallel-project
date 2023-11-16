extern "C" {
#include "pipeline.h"
}
#include <cuda_runtime_api.h>
#include <curand.h>
// #include <curand_kernel.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
// #include "vec3.h"

typedef struct {
  double x;
  double y;
  double z;
} Vec3;

#define MR_MULTIPLIER 279470273
#define MR_MODULUS 4294967291U
#define MR_DIVISOR ((double)4294967291U)
__device__ unsigned my_rand(unsigned *seed_p) {
  long long z = *seed_p;
  z *= MR_MULTIPLIER;
  z %= MR_MODULUS;
  *seed_p = z;
  return *seed_p;
}

__device__ double my_drand(unsigned *seed_p) {
  unsigned x = my_rand(seed_p);
  double y = x / MR_DIVISOR;
  return y;
}

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
__device__ double sq_len_v(Vec3 *vec) {
  return vec->x * vec->x + vec->y * vec->y + vec->z * vec->z;
}
__device__ double len_v(Vec3 *vec) {
  return sqrt(vec->x * vec->x + vec->y * vec->y + vec->z * vec->z);
}
__device__ void normalize_v(Vec3 *vec) { divide_v(vec, len_v(vec)); }

__device__ int ray_intersect(Vec3 *o, Vec3 *d, Object *object,
                             Vec3 *intersection, Vec3 *normal, double *best_t) {
  Vec3 oc;
  oc.x = o->x - object->x;
  oc.y = o->y - object->y;
  oc.z = o->z - object->z;

  double a = dot_v(d, d);
  double b = 2.0 * dot_v(&oc, d);
  double c = dot_v(&oc, &oc) - object->radius * object->radius;
  double discriminant = b * b - 4 * a * c;

  if (discriminant < 0) {
    return 0; // No intersection
  } else {
    double sqrt_discr = sqrt(discriminant);
    double t1 = (-b - sqrt_discr) / (2 * a);
    double t2 = (-b + sqrt_discr) / (2 * a);

    if (t1 > 0 || t2 > 0) {
      // Intersection found
      *best_t = (t1 < t2) ? t1 : t2;

      intersection->x = o->x + *best_t * d->x;
      intersection->y = o->y + *best_t * d->y;
      intersection->z = o->z + *best_t * d->z;

      normal->x = (intersection->x - object->x) / object->radius;
      normal->y = (intersection->y - object->y) / object->radius;
      normal->z = (intersection->z - object->z) / object->radius;

      return 1;
    }
  }

  return 0; // No intersection
}
__device__ int find_closest_hit(Vec3 *o, Vec3 *d, Object *objects, int count,
                                Vec3 *intersection, Vec3 *normal,
                                int *clostest_object_index) {
  double best_t = 9999;
  Vec3 best_inter, best_normal;
  for (int i = 0; i < count; i++) {
    Vec3 intersection, normal;
    double t;
    if (ray_intersect(o, d, &(objects[i]), &intersection, &normal, &t)) {
      if (t < best_t) {
        best_t = t;
        best_inter = intersection;
        best_normal = normal;
        *clostest_object_index = i;
      }
    }
  }
  if (best_t < 9999) {
    *intersection = best_inter;
    *normal = best_normal;
    return 1;
  } else
    return 0;
}
__device__ void reflect(Vec3 *dir, Vec3 *normal, Vec3 *reflected_dir) {
  double dot_product = dot_v(dir, normal) * 2.0;

  reflected_dir->x = dir->x - dot_product * normal->x;
  reflected_dir->y = dir->y - dot_product * normal->y;
  reflected_dir->z = dir->z - dot_product * normal->z;
}

// Rotate a direction vector around the x axis by angle degrees
__device__ void rotateX(Vec3 *dir, double angle) {
  double cosA = cos(angle * M_PI / 180.0);
  double sinA = sin(angle * M_PI / 180.0);

  double newY = dir->y * cosA - dir->z * sinA;
  double newZ = dir->y * sinA + dir->z * cosA;

  dir->y = newY;
  dir->z = newZ;
}

// Rotate a direction vector around the y axis by angle degrees
__device__ void rotateY(Vec3 *dir, double angle) {
  double cosA = cos(angle * M_PI / 180.0);
  double sinA = sin(angle * M_PI / 180.0);

  double newX = dir->x * cosA + dir->z * sinA;
  double newZ = -dir->x * sinA + dir->z * cosA;

  dir->x = newX;
  dir->z = newZ;
}

// Rotate a direction vector around the z axis by angle degrees
__device__ void rotateZ(Vec3 *dir, double angle) {
  double cosA = cos(angle * M_PI / 180.0);
  double sinA = sin(angle * M_PI / 180.0);

  double newX = dir->x * cosA - dir->y * sinA;
  double newY = dir->x * sinA + dir->y * cosA;

  dir->x = newX;
  dir->y = newY;
}

// Rotate a direction vector around the x, y, and z axes by given angles
__device__ void rotateDirection(Vec3 *dir, double angleX, double angleY,
                                double angleZ) {
  rotateX(dir, angleX);
  rotateY(dir, angleY);
  rotateZ(dir, angleZ);
}

__device__ float random_normal(unsigned *seed) {
  float theta = 2 * 3.1415926 * my_rand(seed);
  float rho = sqrt(-2 * log((double)my_rand(seed)));
  return rho * cos(theta);
}
__device__ void random_direction(Vec3 *target, unsigned *seed) {
  target->x = random_normal(seed);
  target->y = random_normal(seed);
  target->z = random_normal(seed);
  normalize_v(target);
}
__device__ void random_direction_hemi(Vec3 *target, Vec3 *normal,
                                      unsigned *seed) {
  random_direction(target, seed);
  if (dot_v(normal, target))
    mult_v(target, -1);
}

#define PC_VAL 0.005f // Pixel to Coordinate ratio w = -3,3 | h = -2,2

__global__ void test_kernel(Object *objects, int count, UCHAR *r, UCHAR *g,
                            UCHAR *b, int w, int h) {
  int x_p = blockDim.x * blockIdx.x + threadIdx.x;
  int y_p = blockDim.y * blockIdx.y + threadIdx.y;
  if (x_p >= w || y_p >= h)
    return;
  int index = x_p + y_p * w;
  unsigned seed = index;

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

  Vec3 ray_energy = {.x = 0, .y = 0, .z = 0};
  Vec3 ray_color = {.x = 1, .y = 1, .z = 1};

  Vec3 intersection, normal;
  int hit_index;
  int reflect_count = 0;
  while (reflect_count < 5 &&
         find_closest_hit(&r_origin, &r_dir, objects, count, &intersection,
                          &normal, &hit_index)) {
    reflect_count++;
    copy_v(&r_origin, &intersection);
    // reflect(&r_dir, &normal, &r_dir);
    random_direction_hemi(&r_dir, &normal, &seed);

    Object o = objects[hit_index];
    if (hit_index == 0) {
      Vec3 object_enery;
      object_enery.x = 1;
      object_enery.y = 1;
      object_enery.z = 1;

      mult_v(&object_enery, &ray_color);
      add_v(&ray_energy, &object_enery);
    }
    ray_color.x *= o.color.r / 255.0f;
    ray_color.y *= o.color.g / 255.0f;
    ray_color.z *= o.color.b / 255.0f;
  }

  r[index] = ray_energy.x * 255.0;
  g[index] = ray_energy.y * 255.0;
  b[index] = ray_energy.z * 255.0;
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
