#include <stdlib.h>
#include <stdio.h>
#include "likwidinc.h"

#ifdef LIKWID_PERFMON
#include <likwid-marker.h>
#else
#define LIKWID_MARKER_INIT
#define LIKWID_MARKER_THREADINIT
#define LIKWID_MARKER_SWITCH
#define LIKWID_MARKER_REGISTER(regionTag)
#define LIKWID_MARKER_START(regionTag)
#define LIKWID_MARKER_STOP(regionTag)
#define LIKWID_MARKER_CLOSE
#define LIKWID_MARKER_GET(regionTag, nevents, events, time, count)
#endif

void perform_computation_one(double *a,
                             const double *b,
                             const size_t N) {
  LIKWID_MARKER_START("computation_one");
  for (size_t i = 1; i < N; i++) {
    for (size_t j = 0; j < i; j++) {
      a[i] += b[j];
    }
  }
  LIKWID_MARKER_STOP("computation_one");
}

double perform_computation_two(const double *a,
                               const double *b,
                               const size_t N) {
  double c = 0;
  LIKWID_MARKER_START("computation_two");
  for (size_t i = 1; i < N; i+=3) {
    for (size_t j = 0; j < i; j++) {
      c += a[j] * b[j];
    }
  }
  LIKWID_MARKER_STOP("computation_two");
  return c;
}

int main(int argc, char **argv) {

  /* Parse input. */
  size_t N = 1000;
  if (argc >= 2) {
    sscanf(argv[1], "%zu", &N);
  }
  if (argc > 2)
    printf("WARNING: additional parameters ignored.\n");

  printf("Using N = %lu\n", N);

  /* Allocate arrays. */
  double *a = malloc(N * sizeof *a);
  double *b = malloc(N * sizeof *b);

  /* Initialise vectors. */
  for (size_t i = 0; i < N; i++) {
    a[i] = 0;
    b[i] = (rand() / (double)RAND_MAX) - 0.5;
  }

  LIKWID_MARKER_INIT;
  LIKWID_MARKER_THREADINIT;
  /* Run subfunctions and print results. */
  perform_computation_one(a, b, N);
  printf("First result is %.4e\n", a[N-1]);

  double c = perform_computation_two(a, b, N);
  printf("Second result is %.4e\n", c);
  LIKWID_MARKER_CLOSE;
  return 0;
}
