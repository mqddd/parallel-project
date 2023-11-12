#include "frame.h"
#include "scene.h"

typedef struct PipelineSetting {
  int debug;
  int save;
  char *out_file;
} PipelineSetting;

typedef Frame *(*Renderer)(Scene *, PipelineSetting);
