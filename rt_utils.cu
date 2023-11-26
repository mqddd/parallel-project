extern "C" {
#include "pipeline.h"
}
#include "cuda_vec.cu"

// #define MR_MULTIPLIER 279470273
// #define MR_MODULUS 4294967291U
// #define MR_DIVISOR ((float)4294967291U)
// __device__ __forceinline__ unsigned my_rand(unsigned *seed_p) {
//   long long z = *seed_p;
//   z *= MR_MULTIPLIER;
//   z %= MR_MODULUS;
//   *seed_p = z;
//   return *seed_p;
// }

// __device__ __forceinline__ float my_drand(unsigned *seed_p) {
//   unsigned x = my_rand(seed_p);
//   float y = x / MR_DIVISOR;
//   return y * 0.99 + 0.01;
// }

__device__ __forceinline__ unsigned my_rand(unsigned *seed_p) {
  *seed_p = *seed_p * 747796405 + 2891336453;
  unsigned result = ((*seed_p >> ((*seed_p >> 28) + 4)) ^ *seed_p) * 277803737;
  result = (result >> 22) ^ result;
  return result;
}

__device__ __forceinline__ float my_drand(unsigned *seed_p) {
  return my_rand(seed_p) / 4294967295.0;
}

Scene *sample_scene_cuda() {
  Object objects[] = {
      // {.pos = {.a = 150, .b = 50, .c = 300},
      //  .radius = 100,
      //  .material =
      //      {
      //          .color = {.a = 1, .b = 1, .c = 0.05},
      //          .emission_color = {.a = 0.863, .b = 0.949, .c = 0.961},
      //          .emission_strength = 300,
      //      }},
      {.pos = {.a = -75, .b = 25, .c = 200},
       .radius = 50,
       .material =
           {
               .color = {.a = 0.05, .b = 0.05, .c = 0.05},
               .emission_color = {.a = 1, .b = 1, .c = 0.6},
               .emission_strength = 3,
           }},
      {.pos = {.a = 0, .b = -1000, .c = 50},
       .radius = 1000,
       .material = {.color = {.a = 0.4, .b = 0.4, .c = 0.4}}},
      {.pos = {.a = -3, .b = 1, .c = 20},
       .radius = 1,
       .material = {.color = {.a = 1, .b = 0.05, .c = 0.05},
                    .specular_rate = 0.9}},
      {.pos = {.a = 0, .b = 5, .c = 40},
       .radius = 10,
       .material = {.color = {.a = 0.9, .b = 0.9, .c = 1}, .specular_rate = 1}},
      {.pos = {.a = 3, .b = 1, .c = 20},
       .radius = 1,
       .material = {.color = {.a = 1, .b = 0, .c = 0},
                    .emission_color = {.a = 0.1, .b = 0.1, .c = 1},
                    .emission_strength = 10}},
      {.pos = {.a = 10, .b = 1, .c = 30},
       .radius = 1,
       .material = {.color = {.a = 0.7, .b = 0.7, .c = 1}}},
      {.pos = {.a = 2, .b = 10, .c = 20},
       .radius = 4,
       .material = {.color = {.a = 1, .b = 0.8, .c = 1}, .specular_rate = 1}},
      {.pos = {.a = 10, .b = 5, .c = 30},
       .radius = 1,
       .material = {.color = {.a = 0.05, .b = 0.8, .c = 0.8}}},
      {.pos = {.a = 10, .b = 1, .c = 25},
       .radius = 0.2,
       .material = {.color = {.a = 1, .b = 0.05, .c = 0.05},
                    .emission_color = {.a = 1, .b = 0.2, .c = 0.2},
                    .emission_strength = 50}},
      {.pos = {.a = 400, .b = 200, .c = 800},
       .radius = 100,
       .material = {.color = {.a = 1, .b = 1, .c = 1},
                    .emission_color = {.a = 1, .b = 1, .c = 0.2},
                    .emission_strength = 50}},
  };
  int count = sizeof(objects) / sizeof(objects[0]);

  return create_scene(objects, count);
}

__device__ __forceinline__ int ray_intersect(Vec3 *o, Vec3 *d,
                                             const Object *object,
                                             Vec3 *intersection, Vec3 *normal,
                                             float *dst) {
  Vec3 oc;
  oc.x = o->x - object->pos.a;
  oc.y = o->y - object->pos.b;
  oc.z = o->z - object->pos.c;

  float a = dot_v(d, d);
  float b = 2.0 * dot_v(&oc, d);
  float c = dot_v(&oc, &oc) - object->radius * object->radius;
  float discriminant = b * b - 4 * a * c;

  if (discriminant < 0 || b > 0) {
    return 0;
  } else {
    float sqrt_discr = sqrt(discriminant);
    *dst = (-b - sqrt_discr) / (2 * a);
    if (*dst > 0) {

      intersection->x = o->x + *dst * d->x;
      intersection->y = o->y + *dst * d->y;
      intersection->z = o->z + *dst * d->z;

      normal->x = (intersection->x - object->pos.a) / object->radius;
      normal->y = (intersection->y - object->pos.b) / object->radius;
      normal->z = (intersection->z - object->pos.c) / object->radius;

      return 1;
    }
  }

  return 0;
}
__device__ __forceinline__ int
find_closest_hit(Vec3 *o, Vec3 *d, const Object *objects, int count,
                 int last_hit_index, Vec3 *intersection, Vec3 *normal,
                 int *clostest_object_index) {
  float current_dst, best_dst = 9999999;
  Vec3 current_intersection, current_normal;
  for (int i = 0; i < count; i++) {
    if (last_hit_index == i)
      continue;
    if (ray_intersect(o, d, &(objects[i]), &current_intersection,
                      &current_normal, &current_dst)) {
      if (current_dst < best_dst) {
        best_dst = current_dst;
        copy_v(intersection, &current_intersection);
        copy_v(normal, &current_normal);
        *clostest_object_index = i;
      }
    }
  }
  if (best_dst < 9999999) {
    return 1;
  } else
    return 0;
}
__device__ __forceinline__ float random_normal(unsigned *seed) {
  float theta = 2 * 3.1415926 * my_drand(seed);
  float rho = sqrt(-2 * log(my_drand(seed)));
  // float rho = 1;
  return rho * cos(theta);
}
// __device__ __forceinline__ void random_direction(Vec3 *target, unsigned
// *seed) {
//   target->x = random_normal(seed);
//   target->y = random_normal(seed);
//   target->z = random_normal(seed);
//   normalize_v(target);
// }
// __device__ __forceinline__ void
// random_direction_hemi(Vec3 *target, Vec3 *normal, unsigned *seed) {
//   random_direction(target, seed);
//   if (dot_v(normal, target) < 0)
//     mult_v(target, -1);
// }
__device__ __forceinline__ void random_direction_hemi_and_lerp(Vec3 *target,
                                                               Vec3 *normal,
                                                               unsigned *seed,
                                                               float lerp) {
  float rx = random_normal(seed) + normal->x;
  float ry = random_normal(seed) + normal->y;
  float rz = random_normal(seed) + normal->z;
  // float dot = rx * normal->x + ry * normal->y + rz * normal->z;
  // rx = dot < 0 ? -rx : rx;
  // ry = dot < 0 ? -ry : ry;
  // rz = dot < 0 ? -rz : rz;
  target->x = target->x * (1.0 - lerp) + (rx * lerp);
  target->y = target->y * (1.0 - lerp) + (ry * lerp);
  target->z = target->z * (1.0 - lerp) + (rz * lerp);
}

__device__ __forceinline__ void
move_point_randomly_in_circle(Vec3 *target, unsigned *seed, float radius) {
  float theta = 2 * 3.1415926 * my_drand(seed);
  float rho = sqrt(my_drand(seed)) * radius;
  target->x += rho * cos(theta);
  target->y += rho * sin(theta);
}

__device__ __forceinline__ float lerp(float a, float b, float f) {
  return a * (1.0 - f) + (b * f);
}