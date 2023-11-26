#include "frame.h"
#include "scene.h"

typedef struct PipelineSetting {
  int width;
  int height;
  int ray_per_pixel;
  int cuda_coercion_rate;
  int debug;
  int save;
  char *out_file;
} PipelineSetting;

typedef void (*Renderer)(Scene *, Frame *, PipelineSetting);

void pipeline(Scene *scene, PipelineSetting setting, Renderer renderer);
