#include <stdio.h>
float reduce(int N, const float *restrict a)
{
  float c = 0;
  for (int i = 0; i < N; i++)
    c += a[i];
  return c;
}


int main() {

return 0;
}

