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

