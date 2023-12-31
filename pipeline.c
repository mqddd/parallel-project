#include "pipeline.h"
#include <stdio.h>
#include <sys/time.h>
#include <string.h>

void pipeline(Scene *scene, PipelineSetting setting, Renderer renderer, int render_type) {
  struct timeval start, stop;
  double time;
  gettimeofday(&start, NULL);

  Frame *frame = create_frame(setting.width, setting.height);
  renderer(scene, frame, setting);

  if (setting.debug) {
    gettimeofday(&stop, NULL);
    time = (stop.tv_sec - start.tv_sec) * 1000 +
           (double)(stop.tv_usec - start.tv_usec) / 1000;
    if (render_type == 1) {
      printf("CUDA render took %.4f ms \n\n", time);
    } else {
      printf("OMP render took %.4f ms \n\n", time);
    }
  }

  if (setting.save && setting.out_file) {
    char* err = save_frame(frame, setting.out_file);
    if (err)
      printf("Save error: %s\n", err);
  }
  free_frame(frame);
}