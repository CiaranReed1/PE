#include <stdlib.h>
#include <stdio.h>

#ifdef LIKWID_PERFMON
#include <likwid.h>
#else
#define LIKWID_MARKER_START(a) do { (void)a; } while (0)
#define LIKWID_MARKER_STOP(a) do { (void)a; } while (0)
#define LIKWID_MARKER_INIT do { } while (0)
#define LIKWID_MARKER_THREADINIT do { } while (0)
#define LIKWID_MARKER_CLOSE do { } while (0)
#define LIKWID_MARKER_REGISTER(a) do { (void)a; } while (0)
#endif  /* LIKWID_PERFMON */


void myloop(int n, const double * restrict x, double * restrict y, double * restrict z){
LIKWID_MARKER_START("MYLOOP");
for (int i = 0; i < n; i++) {
  y[i] = x[i] * 2.0 + z[i];
  z[i] = z[i] + y[i];
}
LIKWID_MARKER_STOP("MYLOOP");
}

int main(int argc, char **argv) {
    double *a = NULL;
    double *b = NULL;
    double *c = NULL;
    int n;
    LIKWID_MARKER_INIT;


    n = atoi(argv[1]);
    if (posix_memalign((void**)&a, 64, (n+1) * sizeof(*a)))
        return 1;
    if (posix_memalign((void**)&b, 64, (n+1) * sizeof(*b)))
        return 1;
    if (posix_memalign((void**)&c, 64, (n+1) * sizeof(*c)))
        return 1;

    for (int i = 0; i < n; i++) {
        a[i] = i;
        b[i] = i-10;
        c[i] = 0;
    }

    myloop(n, a, b, c);

    LIKWID_MARKER_CLOSE;
    free(a);
    free(b);
    free(c);
    return 0; 
}