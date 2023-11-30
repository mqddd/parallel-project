typedef struct {
  float a;
  float b;
  float c;
} Float3;

typedef struct {
  Float3 color;
  Float3 emission_color;
  float specular_rate;
  float emission_strength;
} Material;

typedef struct {
  Float3 pos;
  float radius;
  Material material;
} Object;

typedef struct Scene {
  int count;
  Object *objects;
} Scene;

Scene *create_scene(Object *objects, int count);

void free_scene(Scene *scene);

Scene *sample_scene1();