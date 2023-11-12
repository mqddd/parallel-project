#ifndef UCHAR
#define UCHAR unsigned char
#endif /* ifndef UCHAR */

typedef struct Frame {
  int width;
  int height;
  UCHAR *r;
  UCHAR *g;
  UCHAR *b;
} Frame;

void save_frame(Frame *frame, char *file);

void free_frame(Frame *frame);
