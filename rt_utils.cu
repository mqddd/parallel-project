extern "C" {
#include "pipeline.h"
}
#include "cuda_vec.cu"

#define MR_MULTIPLIER 279470273
#define MR_MODULUS 4294967291U
#define MR_DIVISOR ((double)4294967291U)
__device__  __host__ unsigned my_rand(unsigned *seed_p) {
  long long z = *seed_p;
  z *= MR_MULTIPLIER;
  z %= MR_MODULUS;
  *seed_p = z;
  return *seed_p;
}

Scene *sample_scene_cuda() {
  Object objects[] = {
      {.pos = {.a = -75, .b = 75, .c = 200},
       .radius = 100,
       .material =
           {
               .color = {.a = 0.05, .b = 0.05, .c = 0.05},
               .emission_color = {.a = 1, .b = 1, .c = 1},
               .emission_strength = 3,
           }},
      {.pos = {.a = 0, .b = -1000, .c = 50},
       .radius = 1000,
       .material = {.color = {.a = 0.4, .b = 0.4, .c = 0.4}}},
      {.pos = {.a = -3, .b = 1, .c = 20},
       .radius = 1,
       .material = {.color = {.a = 1, .b = 0.05, .c = 0.05}}},
      {.pos = {.a = 0, .b = 5, .c = 40},
       .radius = 10,
       .material = {.color = {.a = 0.9, .b = 0.9, .c = 1}, .specular_rate = 1}},
      {.pos = {.a = 3, .b = 1, .c = 20},
       .radius = 1,
       .material = {.color = {.a = 0, .b = 0, .c = 1},
                    .emission_color = {.a = 0.1, .b = 0.1, .c = 1},
                    .emission_strength = 1}},
      {.pos = {.a = 10, .b = 1, .c = 30},
       .radius = 1,
       .material = {.color = {.a = 0.4, .b = 0.4, .c = 0.8}}},
      {.pos = {.a = 2, .b = 10, .c = 20},
       .radius = 4,
       .material = {.color = {.a = 1, .b = 0.8, .c = 1}, .specular_rate = 1}},
      {.pos = {.a = 10, .b = 5, .c = 30},
       .radius = 1,
       .material = {.color = {.a = 0.05, .b = 0.4, .c = 0.4}}},
      {.pos = {.a = 10, .b = 1, .c = 25},
       .radius = 0.2,
       .material = {.color = {.a = 1, .b = 0.05, .c = 0.05},
                    .emission_color = {.a = 1, .b = 1, .c = 1},
                    .emission_strength = 50}},
  };
  int count = sizeof(objects) / sizeof(objects[0]);

  return create_scene(objects, count);
}

__device__  __host__ double my_drand(unsigned *seed_p) {
  unsigned x = my_rand(seed_p);
  double y = x / MR_DIVISOR;
  return y * 0.99 + 0.01;
}

__device__  __host__ int ray_intersect(Vec3 *o, Vec3 *d, Object *object,
                             Vec3 *intersection, Vec3 *normal, double *dst) {
  Vec3 oc;
  oc.x = o->x - object->pos.a;
  oc.y = o->y - object->pos.b;
  oc.z = o->z - object->pos.c;

  double a = dot_v(d, d);
  double b = 2.0 * dot_v(&oc, d);
  double c = dot_v(&oc, &oc) - object->radius * object->radius;
  double discriminant = b * b - 4 * a * c;

  if (discriminant < 0 || b > 0) {
    return 0;
  } else {
    double sqrt_discr = sqrt(discriminant);
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
__device__  __host__ int find_closest_hit(Vec3 *o, Vec3 *d, Object *objects, int count,
                                int last_hit_index, Vec3 *intersection,
                                Vec3 *normal, int *clostest_object_index) {
  double best_dst = 9999999;
  Vec3 best_inter, best_normal;
  for (int i = 0; i < count; i++) {
    if (last_hit_index == i)
      continue;
    double t;
    if (ray_intersect(o, d, &(objects[i]), intersection, normal, &t)) {
      if (t < best_dst) {
        best_dst = t;
        copy_v(&best_inter, intersection);
        copy_v(&best_normal, normal);
        *clostest_object_index = i;
      }
    }
  }
  if (best_dst < 9999999) {
    copy_v(intersection, &best_inter);
    copy_v(normal, &best_normal);
    normalize_v(normal);
    // Vec3 copy;
    // copy_v(&copy, normal);
    // double len = len_v(normal);
    // if (len > 1.001 || len < 0.999) {
    //   divide_v(normal, len);
    // }
    return 1;
  } else
    return 0;
}
__device__  __host__ float random_normal(unsigned *seed) {
  float theta = 2 * 3.1415926 * my_drand(seed);
  float rho = sqrt(-2 * log((double)my_drand(seed)));
  return rho * cos(theta);
}
__device__  __host__ void random_direction(Vec3 *target, unsigned *seed) {
  target->x = random_normal(seed);
  target->y = random_normal(seed);
  target->z = random_normal(seed);
  normalize_v(target);
}
__device__  __host__ void random_direction_hemi(Vec3 *target, Vec3 *normal,
                                      unsigned *seed) {
  random_direction(target, seed);
  if (dot_v(normal, target) < 0)
    mult_v(target, -1);
}
__device__  __host__ void random_direction_hemi_and_lerp(Vec3 *target, Vec3 *normal,
                                               unsigned *seed, double lerp) {
  float rx = random_normal(seed);
  float ry = random_normal(seed);
  float rz = random_normal(seed);
  float dot = rx * normal->x + ry * normal->y + rz * normal->z;
  rx = dot < 0 ? -rx : rx;
  ry = dot < 0 ? -ry : ry;
  rz = dot < 0 ? -rz : rz;
  target->x = target->x * (1.0 - lerp) + (rx * lerp);
  target->y = target->y * (1.0 - lerp) + (ry * lerp);
  target->z = target->z * (1.0 - lerp) + (rz * lerp);
}

__device__ __host__ double lerp(double a, double b, double f) {
  return a * (1.0 - f) + (b * f);
}