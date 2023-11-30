#define assert(message, test) do { if (!(test)) return message; } while (0)

#define run_test(test) do { char *message = test(); tests_run++; \
                                if (message) return message; } while (0)
int tests_run = 0;