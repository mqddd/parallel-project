#include <math.h>

typedef struct {
  float x;
  float y;
  float z;
} Vec3;

void add_v(Vec3 *vec, Vec3 *target) {
  vec->x += target->x;
  vec->y += target->y;
  vec->z += target->z;
}

void _add_v(Vec3 *vec, int v) {
  vec->x += v;
  vec->y += v;
  vec->z += v;
}

void sub_v(Vec3 *vec, Vec3 *target) {
  vec->x -= target->x;
  vec->y -= target->y;
  vec->z -= target->z;
}

void _sub_v(Vec3 *vec, int v) {
  vec->x -= v;
  vec->y -= v;
  vec->z -= v;
}

void mult_v(Vec3 *vec, Vec3 *target) {
  vec->x *= target->x;
  vec->y *= target->y;
  vec->z *= target->z;
}

void _mult_v(Vec3 *vec, int v) {
  vec->x *= v;
  vec->y *= v;
  vec->z *= v;
}

void divide_v(Vec3 *vec, int v) {
  vec->x /= v;
  vec->y /= v;
  vec->z /= v;
}

float dot_v(Vec3 *vec, Vec3 *target) {
  return vec->x * target->x + vec->y * target->y + vec->z * target->z;
}

void cross_v(Vec3 *vec, Vec3 *target) {
  float x = vec->x;
  float y = vec->y;
  float z = vec->z;
  vec->x = y * target->z - z * target->y;
  vec->y = -x * target->z - z * target->x;
  vec->z = x * target->y - y * target->x;
}

void copy_v(Vec3 *vec, Vec3 *target) {
  vec->x = target->x;
  vec->y = target->y;
  vec->z = target->z;
}

float sq_len_v(Vec3 *vec) {
  return vec->x * vec->x + vec->y * vec->y + vec->z * vec->z;
}

float len_v(Vec3 *vec) {
  return sqrt(vec->x * vec->x + vec->y * vec->y + vec->z * vec->z);
}

void normalize_v(Vec3 *vec) { divide_v(vec, len_v(vec)); }

void reflect(Vec3 *dir, Vec3 *normal, Vec3 *reflected_dir) {
  float dot_product = dot_v(dir, normal) * 2.0;

  reflected_dir->x = dir->x - dot_product * normal->x;
  reflected_dir->y = dir->y - dot_product * normal->y;
  reflected_dir->z = dir->z - dot_product * normal->z;
}

void rotateX(Vec3 *dir, float angle) {
  float cosA = cos(angle * M_PI / 180.0);
  float sinA = sin(angle * M_PI / 180.0);

  float newY = dir->y * cosA - dir->z * sinA;
  float newZ = dir->y * sinA + dir->z * cosA;

  dir->y = newY;
  dir->z = newZ;
}

void rotateY(Vec3 *dir, float angle) {
  float cosA = cos(angle * M_PI / 180.0);
  float sinA = sin(angle * M_PI / 180.0);

  float newX = dir->x * cosA + dir->z * sinA;
  float newZ = -dir->x * sinA + dir->z * cosA;

  dir->x = newX;
  dir->z = newZ;
}

void rotateZ(Vec3 *dir, float angle) {
  float cosA = cos(angle * M_PI / 180.0);
  float sinA = sin(angle * M_PI / 180.0);

  float newX = dir->x * cosA - dir->y * sinA;
  float newY = dir->x * sinA + dir->y * cosA;

  dir->x = newX;
  dir->y = newY;
}

void rotateDirection(Vec3 *dir, float angleX, float angleY,
                                float angleZ) {
  rotateX(dir, angleX);
  rotateY(dir, angleY);
  rotateZ(dir, angleZ);
}