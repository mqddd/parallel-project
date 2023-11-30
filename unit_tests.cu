#include <stdlib.h>
#include <stdio.h>
#include "test.h"
#include "cuda_vec.cu"

static char* test_add_v() 
{

}

static char* test_add_v() 
{

}

static char* test_sub_v() 
{

}

static char* test_sub_v() 
{

}

static char* test_mult_v() 
{

}

static char* test_mult_v() 
{

}

static char* test_divide_v() 
{

}

static char* test_dot_v() 
{

}

static char* test_cross_v() 
{

}

static char* test_copy_v() 
{

}

static char* testSq_len_v() 
{

}

static char* testLen_v() 
{

}

static char* test_normalize_v() 
{ 

}

static char* test_reflect() 
{

}

static char* test_rotateX() 
{

}

static char* test_rotateY() 
{

}

static char* test_rotateZ() 
{

}

static char* test_rotateDirection() 
{

}

static char* test()
{
    Vec3 a = {.x=2, .y=4, .z=8};
    Vec3 b = {.x=2, .y=4, .z=8};
    add_v(&a, &b);
    return "success!";
}

static char* all_tests() 
{
	run_test(test);
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