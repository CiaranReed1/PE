#include <stdlib.h>
#include <stdio.h>

void perform_computation_one(double *a,
                             const double *b,
                             const size_t N) {
  for (size_t i = 1; i < N; i++) {
    for (size_t j = 0; j < i; j++) {
      a[i] += b[j];
    }
  }
}

double perform_computation_two(const double *a,
                               const double *b,
                               const size_t N) {
  double c = 0;
  for (size_t i = 1; i < N; i+=3) {
    for (size_t j = 0; j < i; j++) {
      c += a[j] * b[j];
    }
  }
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

  /* Run subfunctions and print results. */
  perform_computation_one(a, b, N);
  printf("First result is %.4e\n", a[N-1]);

  double c = perform_computation_two(a, b, N);
  printf("Second result is %.4e\n", c);

  return 0;
}
