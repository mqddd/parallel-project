#include "pipeline.h"

void test_renderer(Scene *scene, Frame *frame, PipelineSetting setting) {
  int w = frame->width;
  int h = frame->height;

  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++) {
      int index = x + y * w;
      for (int i = 0; i < scene->count; i++) {
        Object o = scene->objects[i];
        int i_x = x - o.x;
        int i_y = y - o.y;
        if (i_x * i_x + i_y * i_y <= o.radius * o.radius) {
          frame->r[index] = o.color.r;
          frame->g[index] = o.color.g;
          frame->b[index] = o.color.b;
        }
      }
    }
}

int main() {
  PipelineSetting setting = {.width = 1200,
                             .height = 800,
                             .debug = 1,
                             .out_file = "test.bmp",
                             .save = 1};
  Scene *scene = sample_scene1();

  pipeline(scene, setting, test_renderer);

  free_scene(scene);
}
