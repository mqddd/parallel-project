#include "scene.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

Scene *create_scene(Object *objects, int count) {
  Scene *scene = (Scene *)malloc(sizeof(Scene));
  scene->objects = malloc(sizeof(Object) * count);
  memcpy(scene->objects, objects, sizeof(Object) * count);
  scene->count = count;

  return scene;
}

void free_scene(Scene *scene) {
  free(scene->objects);
  free(scene);
}

Scene *sample_scene1() {
  Object objects[] = {{.pos = {.a = 100.0, .b = 200.0, .c = 5.0},
                       .radius = 12.0,
                       .material = {.color = {.a = 120.5, .b = 220.5, .c = 0.5}}}};
  int count = sizeof(objects) / sizeof(objects[0]);

  return create_scene(objects, count);
}

// int main() {
//   Scene * scene = sample_scene1();
//   printf("count %d\n", scene->count);

//   for (int i = 0; i < scene->count; i++){
//     Object o = scene->objects[i];
//     printf("%f %f %f: %f - %d %d %d\n", o.x, o.y, o.z, o.radius, o.color.r,
//     o.color.g, o.color.b);
//   }
// }
