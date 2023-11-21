#include "scene.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

Scene * create_scene(Object* objects, int count) {
  Scene *scene = (Scene *)malloc(sizeof(Scene));
  scene->objects = malloc(sizeof(Object) * count);
  memcpy(scene->objects, objects, sizeof(Object) * count);
  scene->count = count;

  return scene;
}

void free_scene(Scene * scene) {
  free(scene->objects);
  free(scene);
}

Scene *sample_scene1() {
  Object objects[] = {
      {.x = 100, .y = 200, .z = 10, .radius = 145, .color = {.r = 100, .g = 100, .b = 100}},
      {.x = 300, .y = 500, .z = 0, .radius = 60, .color = {.r = 200, .g = 100, .b = 50}},
      {.x = 700, .y = 500, .z = 10, .radius = 20, .color = {.r = 22, .g = 190, .b = 150}},
  };
  int count = sizeof(objects) / sizeof(objects[0]);

  return create_scene(objects, count);
}


// int main() {
//   Scene * scene = sample_scene1();
//   printf("count %d\n", scene->count);

//   for (int i = 0; i < scene->count; i++){
//     Object o = scene->objects[i];
//     printf("%f %f %f: %f - %d %d %d\n", o.x, o.y, o.z, o.radius, o.color.r, o.color.g, o.color.b);
//   }
// }
