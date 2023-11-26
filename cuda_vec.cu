#include <math.h>

typedef struct {
  double x;
  double y;
  double z;
} Vec3;

__device__ __forceinline__ void add_v(Vec3 *vec, Vec3 *target) {
  vec->x += target->x;
  vec->y += target->y;
  vec->z += target->z;
}
__device__ __forceinline__ void add_v(Vec3 *vec, double v) {
  vec->x += v;
  vec->y += v;
  vec->z += v;
}
__device__ __forceinline__ void sub_v(Vec3 *vec, Vec3 *target) {
  vec->x -= target->x;
  vec->y -= target->y;
  vec->z -= target->z;
}
__device__ __forceinline__ void sub_v(Vec3 *vec, double v) {
  vec->x -= v;
  vec->y -= v;
  vec->z -= v;
}
__device__ __forceinline__ void mult_v(Vec3 *vec, Vec3 *target) {
  vec->x *= target->x;
  vec->y *= target->y;
  vec->z *= target->z;
}
__device__ __forceinline__ void mult_v(Vec3 *vec, double v) {
  vec->x *= v;
  vec->y *= v;
  vec->z *= v;
}
__device__ __forceinline__ void divide_v(Vec3 *vec, double v) {
  vec->x /= v;
  vec->y /= v;
  vec->z /= v;
}
__device__ __forceinline__ double dot_v(Vec3 *vec, Vec3 *target) {
  return vec->x * target->x + vec->y * target->y + vec->z * target->z;
}
__device__ __forceinline__ void cross_v(Vec3 *vec, Vec3 *target) {
  double x = vec->x;
  double y = vec->y;
  double z = vec->z;
  vec->x = y * target->z - z * target->y;
  vec->y = -x * target->z - z * target->x;
  vec->z = x * target->y - y * target->x;
}
__device__ __forceinline__ void copy_v(Vec3 *vec, Vec3 *target) {
  vec->x = target->x;
  vec->y = target->y;
  vec->z = target->z;
}
__device__ __forceinline__ double sq_len_v(Vec3 *vec) {
  return vec->x * vec->x + vec->y * vec->y + vec->z * vec->z;
}
__device__ __forceinline__ double len_v(Vec3 *vec) {
  return sqrt(vec->x * vec->x + vec->y * vec->y + vec->z * vec->z);
}
__device__ __forceinline__ void normalize_v(Vec3 *vec) {
  divide_v(vec, len_v(vec));
}

__device__ __forceinline__ void reflect(Vec3 *dir, Vec3 *normal,
                                        Vec3 *reflected_dir) {
  double dot_product = dot_v(dir, normal) * 2.0;

  reflected_dir->x = dir->x - dot_product * normal->x;
  reflected_dir->y = dir->y - dot_product * normal->y;
  reflected_dir->z = dir->z - dot_product * normal->z;
}

__device__ __forceinline__ void rotateX(Vec3 *dir, double angle) {
  double cosA = cos(angle * M_PI / 180.0);
  double sinA = sin(angle * M_PI / 180.0);

  double newY = dir->y * cosA - dir->z * sinA;
  double newZ = dir->y * sinA + dir->z * cosA;

  dir->y = newY;
  dir->z = newZ;
}
__device__ __forceinline__ void rotateY(Vec3 *dir, double angle) {
  double cosA = cos(angle * M_PI / 180.0);
  double sinA = sin(angle * M_PI / 180.0);

  double newX = dir->x * cosA + dir->z * sinA;
  double newZ = -dir->x * sinA + dir->z * cosA;

  dir->x = newX;
  dir->z = newZ;
}
__device__ __forceinline__ void rotateZ(Vec3 *dir, double angle) {
  double cosA = cos(angle * M_PI / 180.0);
  double sinA = sin(angle * M_PI / 180.0);

  double newX = dir->x * cosA - dir->y * sinA;
  double newY = dir->x * sinA + dir->y * cosA;

  dir->x = newX;
  dir->y = newY;
}
__device__ __forceinline__ void rotateDirection(Vec3 *dir, double angleX,
                                                double angleY, double angleZ) {
  rotateX(dir, angleX);
  rotateY(dir, angleY);
  rotateZ(dir, angleZ);
}