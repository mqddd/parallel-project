typedef struct Color {
  int r;
  int g;
  int b;
} Color;

typedef struct Object {
  float x;
  float y;
  float z;

  float radius;
  Color color;
} Object;

typedef struct Scene {
  int count;
  Object *objects;
} Scene;

Scene *create_scene(Object *objects, int count);

void free_scene(Scene *scene);

Scene *sample_scene1();
