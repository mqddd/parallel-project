#include <math.h>

typedef struct {
  float x;
  float y;
  float z;
} Vec3;

__device__  __host__  void add_v(Vec3 *vec, Vec3 *target) {
  vec->x += target->x;
  vec->y += target->y;
  vec->z += target->z;
}
__device__  __host__  void add_v(Vec3 *vec, int v) {
  vec->x += v;
  vec->y += v;
  vec->z += v;
}
__device__  __host__  void sub_v(Vec3 *vec, Vec3 *target) {
  vec->x -= target->x;
  vec->y -= target->y;
  vec->z -= target->z;
}
__device__  __host__  void sub_v(Vec3 *vec, int v) {
  vec->x -= v;
  vec->y -= v;
  vec->z -= v;
}
__device__  __host__  void mult_v(Vec3 *vec, Vec3 *target) {
  vec->x *= target->x;
  vec->y *= target->y;
  vec->z *= target->z;
}
__device__  __host__  void mult_v(Vec3 *vec, int v) {
  vec->x *= v;
  vec->y *= v;
  vec->z *= v;
}
__device__  __host__  void divide_v(Vec3 *vec, int v) {
  vec->x /= v;
  vec->y /= v;
  vec->z /= v;
}
__device__  __host__  float dot_v(Vec3 *vec, Vec3 *target) {
  return vec->x * target->x + vec->y * target->y + vec->z * target->z;
}
__device__  __host__  void cross_v(Vec3 *vec, Vec3 *target) {
  float x = vec->x;
  float y = vec->y;
  float z = vec->z;
  vec->x = y * target->z - z * target->y;
  vec->y = -x * target->z - z * target->x;
  vec->z = x * target->y - y * target->x;
}
__device__  __host__  void copy_v(Vec3 *vec, Vec3 *target) {
  vec->x = target->x;
  vec->y = target->y;
  vec->z = target->z;
}
__device__  __host__  float sq_len_v(Vec3 *vec) {
  return vec->x * vec->x + vec->y * vec->y + vec->z * vec->z;
}
__device__  __host__  float len_v(Vec3 *vec) {
  return sqrt(vec->x * vec->x + vec->y * vec->y + vec->z * vec->z);
}
__device__  __host__  void normalize_v(Vec3 *vec) { divide_v(vec, len_v(vec)); }

__device__  __host__  void reflect(Vec3 *dir, Vec3 *normal, Vec3 *reflected_dir) {
  float dot_product = dot_v(dir, normal) * 2.0;

  reflected_dir->x = dir->x - dot_product * normal->x;
  reflected_dir->y = dir->y - dot_product * normal->y;
  reflected_dir->z = dir->z - dot_product * normal->z;
}

__device__  __host__  void rotateX(Vec3 *dir, float angle) {
  float cosA = cos(angle * M_PI / 180.0);
  float sinA = sin(angle * M_PI / 180.0);

  float newY = dir->y * cosA - dir->z * sinA;
  float newZ = dir->y * sinA + dir->z * cosA;

  dir->y = newY;
  dir->z = newZ;
}
__device__  __host__  void rotateY(Vec3 *dir, float angle) {
  float cosA = cos(angle * M_PI / 180.0);
  float sinA = sin(angle * M_PI / 180.0);

  float newX = dir->x * cosA + dir->z * sinA;
  float newZ = -dir->x * sinA + dir->z * cosA;

  dir->x = newX;
  dir->z = newZ;
}
__device__  __host__  void rotateZ(Vec3 *dir, float angle) {
  float cosA = cos(angle * M_PI / 180.0);
  float sinA = sin(angle * M_PI / 180.0);

  float newX = dir->x * cosA - dir->y * sinA;
  float newY = dir->x * sinA + dir->y * cosA;

  dir->x = newX;
  dir->y = newY;
}
__device__  __host__  void rotateDirection(Vec3 *dir, float angleX, float angleY,
                                float angleZ) {
  rotateX(dir, angleX);
  rotateY(dir, angleY);
  rotateZ(dir, angleZ);
}