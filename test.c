#include "pipeline.h"
#include "rt_utils.cu"
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

#define VP_W 0.7f
#define VP_H VP_W * 9 / 16
#define DIAFRAGM 0.002f
#define FOCAL 10
#define R_COUNT 50

void pixel_ray(float x, float y, Vec3 *origin, Vec3 *direction) {
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

void trace_ray(Vec3 *origin, Vec3 *direction, int ray_count,
                          Object *objects, int object_count, Vec3 *ray_energy,
                          unsigned *seed) {
  Vec3 ray_color = {.x = 1, .y = 1, .z = 1};

  Vec3 intersection, normal;
  int hit_index, reflect_count, prev_hit_index = -1;
  Vec3 r_o, r_d, emitted_light;

  Vec3 sky_emitted_light;
  float sky_emitted_light_strength = 0.6;

  for (int i = 0; i < ray_count; i++) {
    ray_color = (Vec3){.x = 1, .y = 1, .z = 1};
    reflect_count = 0;
    copy_v(&r_o, origin);
    copy_v(&r_d, direction);
    prev_hit_index = -1;
    move_point_randomly_in_circle(&r_o, seed, DIAFRAGM / 2);
    r_o.x += my_drand(seed) * 0.001 - 0.0005;
    r_o.y += my_drand(seed) * 0.001 - 0.0005;
    normalize_v(&r_d);
    while (reflect_count < 10) {
      if (find_closest_hit(&r_o, &r_d, objects, object_count, prev_hit_index,
                           &intersection, &normal, &hit_index)) {
        reflect_count++;
        Object *obj = &objects[hit_index];
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
            (1) * sky_emitted_light_strength * ray_color.x;
        sky_emitted_light.y =
            (1) * sky_emitted_light_strength * ray_color.y;
        sky_emitted_light.z =
            (1) * sky_emitted_light_strength * ray_color.z;

        add_v(ray_energy, &sky_emitted_light);
        break;
      }
    }
  }

  ray_energy->x /= ray_count;
  ray_energy->y /= ray_count;
  ray_energy->z /= ray_count;
}

void test_renderer(Scene *scene, Frame *frame, PipelineSetting setting) {
  int w = frame->width;
  int h = frame->height;

  int rays = 1;

  UCHAR *r = (UCHAR *) malloc(sizeof(UCHAR) * w * h * rays);
  UCHAR *g = (UCHAR *) malloc(sizeof(UCHAR) * w * h * rays);
  UCHAR *b = (UCHAR *) malloc(sizeof(UCHAR) * w * h * rays);

  UCHAR *r_out = (UCHAR *) malloc(sizeof(UCHAR) * w * h);
  UCHAR *g_out = (UCHAR *) malloc(sizeof(UCHAR) * w * h);
  UCHAR *b_out = (UCHAR *) malloc(sizeof(UCHAR) * w * h);
  
  // trace rays
  #pragma omp parallel for num_threads(16)
  for (int x_p = 0; x_p < w * rays; x_p++)
  {
    for (int y_p = 0; y_p < h; y_p++)
    {
      int x_p_temp = x_p / rays;

      int index = y_p * w + x_p;
      unsigned int seed = index + 10;

      float x = ((x_p_temp - w / 2.0) / w) * VP_W * FOCAL * 2,
             y = -((y_p - h / 2.0) / h) * VP_H * FOCAL * 2;
      
      Vec3 r_origin;
      Vec3 r_dir;
      pixel_ray(x, y, &r_origin, &r_dir);

      Vec3 ray_energy = {.x = 0, .y = 0, .z = 0};
      trace_ray(&r_origin, &r_dir, R_COUNT, scene->objects, scene->count, &ray_energy, &seed);

      float max_e = fmax(ray_energy.x, fmax(ray_energy.y, ray_energy.z));
      if (max_e > 1) {
        ray_energy.x = (ray_energy.x / max_e) * 1.5;
        ray_energy.y = (ray_energy.y / max_e) * 1.5;
        ray_energy.z = (ray_energy.z / max_e) * 1.5;
      }
      
      r[index] = (ray_energy.x > 1 ? 1 : ray_energy.x) * 255.0;
      g[index] = (ray_energy.y > 1 ? 1 : ray_energy.y) * 255.0;
      b[index] = (ray_energy.z > 1 ? 1 : ray_energy.z) * 255.0; 
    }
  }

  memcpy(frame->r, r, sizeof(UCHAR) * w * h);
  memcpy(frame->g, g, sizeof(UCHAR) * w * h);
  memcpy(frame->b, b, sizeof(UCHAR) * w * h);
}

int main() {
  int width = 1920;
  int height = width * 9 / 16;
  PipelineSetting setting = {.width = width,
                             .height = height,
                             .debug = 1,
                             .out_file = (char *)"test.bmp",
                             .save = 1};
  Scene *scene = sample_scene_2();

  pipeline(scene, setting, test_renderer);

  free_scene(scene);
}
