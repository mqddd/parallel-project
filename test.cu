extern "C" {
#include "pipeline.h"
}
#include <cuda_runtime_api.h>
#include <math.h>
#include <stdbool.h>
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
__device__ void normalize_v(Vec3 *vec) {
  divide_v(vec, len_v(vec));
}

__device__ int ray_intersect_(Vec3 *o, Vec3 *d, Object *object) {
  Vec3 center = {.x = object->x, .y = object->y, .z = object->z};
  Vec3 l;
  copy_v(&l, &center);
  sub_v(&l, o);
  double tc = dot_v(&l, d);
  if (tc < 0)
    return 0;
  double lL2 = sq_len_v(&l);
  double d2 = lL2 - (tc * tc);
  double r2 = object->radius;
  if (o->x > 2.98 && o->y > 1.98) {
    printf("%f %f %f\n", d2, tc*tc, lL2);
  }
  r2 *= r2;
  if (d2 > r2)
    return 0;
  return 1;
}
__device__ int ray_intersect__(Vec3 *o, Vec3 *d, Object *object) {
  Vec3 center = {.x = object->x, .y = object->y, .z = object->z};
  Vec3 l;
  copy_v(&l, o);
  sub_v(&l, &center);
  // double a = dot_v(o, o);
  double b = dot_v(&l, o);
  double c = dot_v(&l, &l) - object->radius * object->radius;
  if (c > 0.0f && b > 0.0f) return 0;
  double discr = b * b - c;

  if (discr < 0.0f) return 0;

  return 1;
}
__device__ int ray_intersect(Vec3 *o, Vec3 *d, Object *object, Vec3 *intersection, Vec3 *normal) {
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
            double t = (t1 < t2) ? t1 : t2;

            intersection->x = o->x + t * d->x;
            intersection->y = o->y + t * d->y;
            intersection->z = o->z + t * d->z;

            normal->x = (intersection->x - object->x) / object->radius;
            normal->y = (intersection->y - object->y) / object->radius;
            normal->z = (intersection->z - object->z) / object->radius;

            return 1;
        }
    }

    return 0; // No intersection
}

#define PC_VAL 0.005f // Pixel to Coordinate ratio w = -3,3 | h = -2,2

__global__ void test_kernel(Object *objects, int count, UCHAR *r, UCHAR *g,
                            UCHAR *b, int w, int h) {
  int x_p = blockDim.x * blockIdx.x + threadIdx.x;
  int y_p = blockDim.y * blockIdx.y + threadIdx.y;
  if (x_p >= w || y_p >= h)
    return;
  int index = x_p + y_p * w;

  double x = (x_p - w / 2.0) * PC_VAL, y = (y_p - h / 2.0) * PC_VAL;

  Vec3 r_origin;
  r_origin.x = 0;
  r_origin.y = 0;
  r_origin.z = 0;
  Vec3 r_dir;
  r_dir.x = x;
  r_dir.y = y;
  r_dir.z = 10.0f;
  divide_v(&r_dir, len_v(&r_dir));

  for (int i = 0; i < count; i++) {
    Object o = objects[i];
    Vec3 intersection, normal;
    if (ray_intersect(&r_origin, &r_dir, &o, &intersection, &normal)) {
      r[index] = o.color.r;
      g[index] = o.color.g;
      b[index] = o.color.b;
      break;
    }
  }
}
__device__ void reflect(Vec3 *dir, Vec3 *normal, Vec3 *reflected_dir) {
    double dot_product = dot_v(dir, normal) * 2.0;

    reflected_dir->x = dir->x - dot_product * normal->x;
    reflected_dir->y = dir->y - dot_product * normal->y;
    reflected_dir->z = dir->z - dot_product * normal->z;
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
