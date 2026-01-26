#define _GNU_SOURCE
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <immintrin.h>
#include <time.h>

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


static void sca(int n, const double * restrict x, double * restrict y, double * restrict z){
LIKWID_MARKER_START("MYLOOP");
#pragma clang loop vectorize(disable)
#pragma novector
for (int i = 0; i < n; i++) {
  y[i] = x[i] * 2.0 + z[i];
  z[i] = z[i] + y[i];
}
LIKWID_MARKER_STOP("MYLOOP");
}

static void sse(int n, const double * restrict x, double * restrict y, double * restrict z){
__m128d x_, y_, z_;
LIKWID_MARKER_START("SSE");
for (int i = 0; i < n; i+= 2) {
    __m128d tmp;
    x_ = _mm_loadu_pd(x + i);
    z_ = _mm_loadu_pd(z + i);
    tmp = _mm_mul_pd(x_, _mm_set1_pd(2.0));
    y_ = _mm_add_pd(z_, tmp);
    _mm_storeu_pd(y + i, y_);
    z_ = _mm_add_pd(z_, y_);
    _mm_storeu_pd(z + i, z_);
}
LIKWID_MARKER_STOP("SSE");
}

static void avx(int n, const double * restrict x, double * restrict y, double * restrict z){
 __m256d x_, y_, z_;
  LIKWID_MARKER_START("AVX");
  for (int i = 0; i < n; i += 4) {
    __m256d tmp;
    x_ = _mm256_loadu_pd(x + i);
    z_ = _mm256_loadu_pd(z + i);
    tmp = _mm256_mul_pd(x_, _mm256_set1_pd(2.0));
    y_ = _mm256_add_pd(z_,tmp);
    _mm256_storeu_pd(y + i, y_);
    z_ = _mm256_add_pd(z_,y_);
    _mm256_storeu_pd(z + i, z_);
  }
  LIKWID_MARKER_STOP("AVX");
}

static void fma(int n, const double * restrict x, double * restrict y, double * restrict z){
 __m256d x_, y_, z_;
  LIKWID_MARKER_START("FMA");
  for (int i = 0; i < n; i += 4) {
    x_ = _mm256_loadu_pd(x + i);
    z_ = _mm256_loadu_pd(z + i);
    y_ = _mm256_fmadd_pd(x_, _mm256_set1_pd(2.0), z_);
    _mm256_storeu_pd(y + i, y_);
    z_ = _mm256_add_pd(z_, y_);
    _mm256_storeu_pd(z + i, z_);
  }
  LIKWID_MARKER_STOP("FMA");
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
    if (!strcmp(argv[2], "sca")) {
        sca(n, a, b, c);
    } 
    else if (!strcmp(argv[2], "sse")) {
        sse(n, a, b, c);
    } 
    else if (!strcmp(argv[2], "avx")) {
        avx(n, a, b, c);
    } 
    else if (!strcmp(argv[2], "fma")) {
        fma(n, a, b, c);
    }
    LIKWID_MARKER_CLOSE;
    free(a);
    free(b);
    free(c);
    return 0; 
}