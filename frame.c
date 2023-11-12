#include "frame.h"
#include "qdbmp.h"
#include <stdio.h>
#include <stdlib.h>

Frame *create_frame(int width, int height) {
  Frame *frame = malloc(sizeof(Frame));
  frame->width = width;
  frame->height = height;
  frame->r = malloc(width * height * sizeof(UCHAR));
  frame->g = malloc(width * height * sizeof(UCHAR));
  frame->b = malloc(width * height * sizeof(UCHAR));
  return frame;
}

void free_frame(Frame *frame) {
  free(frame->r);
  free(frame->g);
  free(frame->b);
  free(frame);
}

void frame_to_bmp(Frame *frame, BMP *bmp) {
  for (int x = 0; x < frame->width; ++x) {
    for (int y = 0; y < frame->height; ++y) {
      int index = x + y * frame->width;
      BMP_SetPixelRGB(bmp, x, y, frame->r[index], frame->g[index],
                      frame->b[index]);
    }
  }
}

char *save_frame(Frame *frame, char *file) {
  BMP *bmp = BMP_Create(frame->width, frame->height, 24);
  frame_to_bmp(frame, bmp);
  BMP_WriteFile(bmp, file);
  BMP_CHECK_ERROR(stdout, "Failed to save.");
}

// int main() {
//   Frame * frame = create_frame(2000, 1440);

//   for (int x = 0; x < frame->width; ++x) {
//     for (int y = 0; y < frame->height; ++y) {
//       int index = x + y * frame->width;
//       frame->r[index] = 255 * (x + y) / (frame->width + frame->height);
//       frame->g[index] = 50;
//       frame->b[index] = 100;
//     }
//   }

//   save_frame(frame, "out.bmp");
//   free_frame(frame);
// }
