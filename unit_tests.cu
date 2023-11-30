#include <stdlib.h>
#include <stdio.h>
#include "test.h"
#include "cuda_vec.cu"

static char* test_add_v() 
{
    Vec3 a = {.x=-25, .y=0, .z=25};
    Vec3 a2 = {.x=a.x, .y=a.y, .z=a.z};
    Vec3 b = {.x=25, .y=25, .z=25};
    Vec3 expected = {.x=0, .y=25, .z=50};
    add_v(&a, &b);
    if (a2.x == a.x || a2.y == a.y || a2.z == a.z) {
        return "1. add_v test failed!\n";
    }
    if (a.x != expected.x || a.y != expected.y || a.z != expected.z) {
        return "2. add_v test failed!\n";
    }

    int c = b.x;
    add_v(&a2, c);
    if (a2.x != a.x || a2.y != a.y || a2.z != a.z) {
        return "3. add_v test failed!\n";
    }

    return 0;
}

static char* test_sub_v() 
{
    Vec3 a = {.x=-25, .y=0, .z=25};
    Vec3 a2 = {.x=a.x, .y=a.y, .z=a.z};
    Vec3 b = {.x=25, .y=25, .z=25};
    Vec3 expected = {.x=-50, .y=-25, .z=0};
    sub_v(&a, &b);
    if (a2.x == a.x || a2.y == a.y || a2.z == a.z) {
        return "1. sub_v test failed!\n";
    }
    if (a.x != expected.x || a.y != expected.y || a.z != expected.z) {
        return "2. sub_v test failed!\n";
    }

    int c = b.x;
    sub_v(&a2, c);
    if (a2.x != a.x || a2.y != a.y || a2.z != a.z) {
        return "3. sub_v test failed!\n";
    }

    return 0;
}

static char* test_mult_v() 
{
    Vec3 a = {.x=-25, .y=0, .z=25};
    Vec3 a2 = {.x=a.x, .y=a.y, .z=a.z};
    Vec3 b = {.x=25, .y=25, .z=25};
    Vec3 expected = {.x=-625, .y=0, .z=625};
    mult_v(&a, &b);
    if (a2.x == a.x || !(a2.y == a.y) || a2.z == a.z) {
        return "1. mul_v test failed!\n";
    }
    if (a.x != expected.x || a.y != expected.y || a.z != expected.z) {
        return "2. mul_v test failed!\n";
    }

    int c = b.x;
    mult_v(&a2, c);
    if (a2.x != a.x || a2.y != a.y || a2.z != a.z) {
        return "3. mul_v test failed!\n";
    }

    return 0;
}

static char* test_divide_v() 
{
    Vec3 a = {.x=-25, .y=0, .z=25};
    Vec3 a2 = {.x=a.x, .y=a.y, .z=a.z};
    int b = 10;
    Vec3 expected = {.x=-2.5, .y=0, .z=2.5};
    divide_v(&a, b);
    if (a2.x == a.x || !(a2.y == a.y) || a2.z == a.z) {
        return "1. divide_v test failed!\n";
    }
    if (a.x != expected.x || a.y != expected.y || a.z != expected.z) {
        return "2. divide_v test failed!\n";
    }

    return 0;
}

static char* test_dot_v() 
{
    Vec3 a = {.x=-25, .y=0, .z=25};
    Vec3 b = {.x=1, .y=25, .z=-1};
    float expected = -50.0;
    float result = dot_v(&a, &b);
    if (result != expected) {
        return "1. dot_v test failed!\n";
    }

    return 0;
}

static char* test_cross_v() 
{
    Vec3 a = {.x=-2, .y=2, .z=4};
    Vec3 b = {.x=10, .y=-2, .z=4};
    Vec3 expected = {.x=16, .y=-32, .z=-16};
    cross_v(&a, &b);
    if (a.x != expected.x || a.y != expected.y || a.z != expected.z) {
        return "1. cross_v test failed!\n";
    }

    return 0;
}

static char* test_copy_v() 
{
    Vec3 a = {.x=-2, .y=2, .z=4};
    Vec3 b;
    copy_v(&b, &a);
    if (a.x != b.x || a.y != b.y || a.z != b.z) {
        return "1. copy_v test failed!\n";
    }

    return 0;
}

static char* test_sq_len_v() 
{
    Vec3 a = {.x=-2, .y=2, .z=4};
    float expected = 24;
    float result = sq_len_v(&a);
    if (result != expected) {
        return "1. sq_len_v test failed!\n";
    }

    return 0;
}

static char* test_len_v() 
{
    Vec3 a = {.x=-2, .y=4, .z=4};
    float expected = 6.0;
    float result = len_v(&a);
    if (result != expected) {
        return "1. len_v test failed!\n";
    }

    return 0;
}

static char* test_normalize_v() 
{ 
    Vec3 a = {.x=-3, .y=6, .z=6};
    Vec3 a2 = {.x=a.x, .y=a.y, .z=a.z};
    divide_v(&a2, len_v(&a2));
    normalize_v(&a);
    if (a2.x != a.x || a2.y != a.y || a2.z != a.z) {
        return "1. normalize_v test failed!\n";
    }

    return 0;
}

static char* test_reflect() 
{
    Vec3 dir = {.x=-2, .y=2, .z=4};
    Vec3 norm = {.x=10, .y=-2, .z=4};
    Vec3 result_reflect;
    reflect(&dir, &norm, &result_reflect);
    mult_v(&norm, (dot_v(&dir, &norm) * 2.0));
    sub_v(&dir, &norm);
    if (dir.x != result_reflect.x || dir.y != result_reflect.y || dir.z != result_reflect.z) {
        return "1. reflect_v test failed!\n";
    }

    return 0;
}

static char* test_rotateX() 
{
    Vec3 dir = {.x=1, .y=1, .z=1};
    float angle = 90.0;
    Vec3 expected = {.x=dir.x, .y=-dir.z, .z=dir.y};
    rotateX(&dir, angle);
    if (dir.x != expected.x || dir.y != expected.y || dir.z != expected.z) {
        return "1. rotateX test failed!\n";
    }

    return 0;
}

static char* test_rotateY() 
{
    Vec3 dir = {.x=1, .y=1, .z=1};
    float angle = 90.0;
    Vec3 expected = {.x=dir.z, .y=dir.y, .z=-dir.x};
    rotateY(&dir, angle);
    if (dir.x != expected.x || dir.y != expected.y || dir.z != expected.z) {
        return "1. rotateY test failed!\n";
    }

    return 0;
}

static char* test_rotateZ() 
{
    Vec3 dir = {.x=1, .y=1, .z=1};
    float angle = 90.0;
    Vec3 expected = {.x=-dir.y, .y=dir.x, .z=dir.z};
    rotateZ(&dir, angle);
    if (dir.x != expected.x || dir.y != expected.y || dir.z != expected.z) {
        return "1. rotateZ test failed!\n";
    }

    return 0;
}

static char* test_rotateDirection() 
{
    Vec3 dir = {.x=1, .y=1, .z=1};
    Vec3 expected = {.x=dir.x, .y=dir.y, .z=dir.z};
    float angleX = 90.0;
    float angleY = 90.0;
    float angleZ = 90.0;
    rotateX(&expected, angleX);
    rotateY(&expected, angleY);
    rotateZ(&expected, angleZ);    
    rotateDirection(&dir, angleX, angleY, angleZ);
    if (dir.x != expected.x || dir.y != expected.y || dir.z != expected.z) {
        return "1. rotateZ test failed!\n";
    }

    return 0;
}

static char* all_tests() 
{
	run_test(test_add_v);
    run_test(test_sub_v);
	run_test(test_mult_v);
	run_test(test_divide_v);
	run_test(test_dot_v);
	run_test(test_cross_v);
	run_test(test_copy_v);
	run_test(test_sq_len_v);
	run_test(test_len_v);
	run_test(test_normalize_v);
	run_test(test_reflect);
	run_test(test_rotateX);
	run_test(test_rotateY);
	run_test(test_rotateZ);
	run_test(test_rotateDirection);

	return 0;
}

int main()
{
    char *result = all_tests();
    if (result != 0) {
        printf("%s\n", result);
    }
    else {
        printf("ALL TESTS PASSED\n");
    }
    printf("Tests run: %d\n", tests_run);
    return 0;
}