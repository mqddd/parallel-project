#include "pipeline.h"
#include "utils.c"
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
// #include <omp.h>

#define VP_W 4.0f
#define VP_H VP_W * 9 / 16
#define DIAFRAGM 0.01f
#define FOCAL 6.0f
#define R_COUNT 50

void pixel_ray(double x, double y, Vec3 *origin, Vec3 *direction) {
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

  Vec3 sky_color, sky_emitted_light;
  float sky_emitted_light_strength = 0.6;

  for (int i = 0; i < ray_count; i++) {
    ray_color = (Vec3){.x = 1, .y = 1, .z = 1};
    reflect_count = 0;
    copy_v(&r_o, origin);
    copy_v(&r_d, direction);
    prev_hit_index = -1;
    r_o.x += my_drand(seed) * DIAFRAGM - DIAFRAGM / 2;
    r_o.y += my_drand(seed) * DIAFRAGM - DIAFRAGM / 2;
    normalize_v(&r_d);
    while (reflect_count < 15) {
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

        emitted_light.x = obj->material.emission_color.a;
        emitted_light.y = obj->material.emission_color.b;
        emitted_light.z = obj->material.emission_color.c;
        _mult_v(&emitted_light, obj->material.emission_strength);

        mult_v(&emitted_light, &ray_color);
        add_v(ray_energy, &emitted_light);
        float max_e = fmax(ray_energy->x, fmax(ray_energy->y, ray_energy->z));

        ray_color.x *= obj->material.color.a;
        ray_color.y *= obj->material.color.b;
        ray_color.z *= obj->material.color.c;
        // float max_c = fmax(ray_color.x, fmax(ray_color.y, ray_color.z));
        // if (max_c > 1) {
        //   ray_color.x /= max_c;
        //   ray_color.y /= max_c;
        //   ray_color.z /= max_c;
        // }
      } else {
        sky_color.x = 0.863f;
        sky_color.y = 0.949f;
        sky_color.z = 0.961f;
        sky_emitted_light.x = sky_emitted_light_strength;
        sky_emitted_light.y = sky_emitted_light_strength;
        sky_emitted_light.z = sky_emitted_light_strength;

        mult_v(&sky_emitted_light, &sky_color);

        mult_v(&sky_emitted_light, &ray_color);
        add_v(ray_energy, &sky_emitted_light);
        float max_e = fmax(ray_energy->x, fmax(ray_energy->y, ray_energy->z));

        break;
      }
    }
  }

  ray_energy->x /= ray_count;
  ray_energy->y /= ray_count;
  ray_energy->z /= ray_count;
  // float max_e = fmax(ray_energy->x, fmax(ray_energy->y, ray_energy->z));
  // if (max_e > 1) {
  //   ray_energy->x /= max_e;
  //   ray_energy->y /= max_e;
  //   ray_energy->z /= max_e;
  // }
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

  // double t = omp_get_wtime();
  
  // trace rays
  // #pragma omp parallel for num_threads(16)
  for (int x_p = 0; x_p < w * rays; x_p++)
  {
    for (int y_p = 0; y_p < h; y_p++)
    {
      int x_p_temp = x_p / rays;

      int index = y_p * w * rays + x_p;
      unsigned int seed = index + 10; // why?

      double x = ((x_p_temp - w / 2.0) / w) * VP_W * 2,
             y = -((y_p - h / 2.0) / h) * VP_H * 2;
      
      Vec3 r_origin;
      Vec3 r_dir;
      pixel_ray(x, y, &r_origin, &r_dir);

      Vec3 ray_energy = {.x = 0, .y = 0, .z = 0};
      trace_ray(&r_origin, &r_dir, R_COUNT, scene->objects, scene->count, &ray_energy, &seed);

      r[index] = (ray_energy.x > 1 ? 1 : ray_energy.x) * 255.0;
      g[index] = (ray_energy.y > 1 ? 1 : ray_energy.y) * 255.0;
      b[index] = (ray_energy.z > 1 ? 1 : ray_energy.z) * 255.0; 
    }
  }

  // t = 1000 * (omp_get_wtime() - t);
  // printf("trace in: %.3f ms\n", t);
  
  // t = omp_get_wtime();

  // average rays
  // #pragma omp parallel for num_threads(16)
  for (int x_p = 0; x_p < w; x_p++)
  {
    for (int y_p = 0; y_p < h; y_p++)
    {
      int index = y_p * w + x_p;
  
      int rp = 0;
      int gp = 0;
      int bp = 0;
      for (int i = 0; i < rays; i++) {
        int in_index = (x_p * rays) + i + y_p * (w * rays);
        rp += r[in_index];
        gp += g[in_index];
        bp += b[in_index];
      }

      r_out[index] = rp / rays;
      g_out[index] = gp / rays;
      b_out[index] = bp / rays;
    }
  }

  // t = 1000 * (omp_get_wtime() - t);
  // printf("average in: %.3f ms\n", t);

  memcpy(frame->r, r_out, sizeof(UCHAR) * w * h);
  memcpy(frame->g, g_out, sizeof(UCHAR) * w * h);
  memcpy(frame->b, b_out, sizeof(UCHAR) * w * h);

  // frame->r = r_out;
  // frame->g = g_out;
  // frame->b = b_out;
  
  // for (int y = 0; y < h; y++)
  //   for (int x = 0; x < w; x++) {
  //     int index = x + y * w;
  //     for (int i = 0; i < scene->count; i++) {
  //       Object o = scene->objects[i];
  //       int i_x = x - o.x;
  //       int i_y = y - o.y;
  //       if (i_x * i_x + i_y * i_y <= o.radius * o.radius) {
  //         frame->r[index] = o.color.r;
  //         frame->g[index] = o.color.g;
  //         frame->b[index] = o.color.b;
  //       }
  //     }
  // }
}

int main() {
  int width = 1280;
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
