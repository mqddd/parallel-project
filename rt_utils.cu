extern "C" {
#include "pipeline.h"
}
#include "cuda_vec.cu"

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
  return y * 0.99 + 0.01;
}

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
    return 0;
  } else {
    double sqrt_discr = sqrt(discriminant);
    double t1 = (-b - sqrt_discr) / (2 * a);
    double t2 = (-b + sqrt_discr) / (2 * a);

    if (t1 > 0 || t2 > 0) {
      *best_t = (t1 < t2) ? t1 : t2;

      intersection->x = o->x + *best_t * d->x;
      intersection->y = o->y + *best_t * d->y;
      intersection->z = o->z + *best_t * d->z;

      normal->x = (intersection->x - object->x) / object->radius;
      normal->y = (intersection->y - object->y) / object->radius;
      normal->z = (intersection->z - object->z) / object->radius;
      normalize_v(normal);

      return 1;
    }
  }

  return 0;
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
__device__ float random_normal(unsigned *seed) {
  float theta = 2 * 3.1415926 * my_drand(seed);
  float rho = sqrt(-2 * log((double)my_drand(seed)));
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
  if (dot_v(normal, target) < 0)
    mult_v(target, -1);
}

__device__ double lerp(double a, double b, double f) {
  return a * (1.0 - f) + (b * f);
}