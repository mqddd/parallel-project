#include <stdlib.h>
#include <stdio.h>
#include "test.h"

__host__ static char* test()
{
    printf("hi\n");
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