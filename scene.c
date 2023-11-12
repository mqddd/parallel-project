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
      {.x = 2, .y = 2, .z = 2, .radius = 2, .color = {.r = 100, .g = 100, .b = 100}},
      {.x = -1, .y = 5, .z = 0, .radius = 1, .color = {.r = 200, .g = 100, .b = 50}},
  };
  int count = sizeof(objects) / sizeof(objects[0]);

  return create_scene(objects, count);
}


int main() {
  Scene * scene = sample_scene1();
  printf("count %d\n", scene->count);

  for (int i = 0; i < scene->count; i++){
    Object o = scene->objects[i];
    printf("%f %f %f: %f - %d %d %d\n", o.x, o.y, o.z, o.radius, o.color.r, o.color.g, o.color.b);
  }
}
