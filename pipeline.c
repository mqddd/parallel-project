#include "pipeline.h"
#include <stdio.h>
#include <sys/time.h>

void pipeline(Scene *scene, PipelineSetting setting, Renderer renderer) {
  struct timeval start, stop;
  double time;

  Frame *frame = renderer(scene, setting);

  if (setting.debug) {
    time = (stop.tv_sec - start.tv_sec) * 1000 +
           (double)(stop.tv_usec - start.tv_usec) / 1000;
    printf("Render took %.4f ms \n\n", time);
  }

  if (setting.save) {
    save_frame(frame, setting.out_file);
  }
  free_frame(frame);
}

int main() {}
