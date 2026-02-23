#include<iostream>
#include<chrono>
//----- Task 1 -----//
// a) Write a CUDA programme that adds the elements of a vector with N elements to each row of a matrix with NxN elements. For this purpose,
//    write a CUDA kernel multi_vector_addition(...) that takes a vector of N doubles and a matrix of NxN doubles as input. You can assume that there are as many threads as elements in the matrix. 
// b) Use CUDA's static shared memory to improve the runtime of the multi_vector_addition(...) kernel. Which data should be stored in shared memory and why? 
// c) Adjust your programme such that the user can specify the size N at runtime. Remember to adjust the grid and block size accordingly.

__global__ void multi_vector_addition(const int N,const double* vector, double* matrix)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < N && y < N){
  matrix[y*N+ x] = matrix[y*N + x] + vector[x];
  }
}


__global__ void multi_vector_addition_shmem(const int N,const double* vector, double* matrix)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  __shared__ double s_v[32]; //this assumes the block size will be fixed at 32 x 32 (not necessarily 16 for the y dim but definitely the x)
  if (x< N){
  s_v[threadIdx.x] = vector[x];
  }
__syncthreads();
  if (x < N && y < N){
  matrix[y*N+ x] = matrix[y*N + x] + s_v[threadIdx.x];
}
}

__global__ void multi_vector_addition_dynamic_shmem(const int N, const double* vector, double* matrix)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  extern __shared__ float s_v[]; //now the blocks can be larger (up to 32 x 32) we want the blocks to be as big as possible to improve shared memory utilisation
  if (x< N){
    s_v[threadIdx.x] = vector[x];
  }
  __syncthreads();
  if (x < N && y < N){
  matrix[y*N+ x] = matrix[y*N + x] + s_v[threadIdx.x];
}
}

//----- Task 2 -----//
//   Consider the subsequent functions f_a(...), f_b(...), f_c(...), f_d(...) and the data dependencies between their parameters as evident in the main(...) function.
// a) Draw the task graph that outlines the dependencies between these functions. Each node in the graph is either a function name, or a parameter. Edges are directed
//    and represent input and output dependencies. Input dependencies 
// b) Rewrite the functions f_a(...), f_b(...), f_c(...), f_d(...) as CUDA kernels.
// c) Complying to data dependencies, launch the kernels concurrently by using CUDA streams.

int f_a(const int a) {
  return a+1;
}

int f_b(const int b) {
  return b+1;
}

int f_c(const int c) {
  return c+1;
}

int f_d(const int a, const int b, const int c) {
  return a+b+c;
}

// ----- Task 3 -----//
// Instrument your code with calls to std::chrono to measure the execution times of 
// your kernels.
//
// As an example:
//   auto t0 = std::chrono::high_resolution_clock::now();
//   my_kernel<<<...>>>(...);
//   cudaDeviceSynchronize(); //<- Why might this be required?
//   auto t1 = std::chrono::high_resolution_clock::now();
//   std::chrono::duration< double > fs = t1 - t0;
//   std::chrono::milliseconds d = std::chrono::duration_cast< std::chrono::milliseconds >( fs );
//   std::cout << fs.count() << "s\n";
//   std::cout << d.count() << "ms\n";

// ----- Task 4 -----//
// Use the Nvidia Nsight Compute Profiler to explore the performance characteristics of the code for Task 1.

//----- Code Template -----//

int main(int argc, char **argv) {

  auto t0 = std::chrono::high_resolution_clock::now();
  auto t1 = std::chrono::high_resolution_clock::now();
  std::chrono::duration< double > fs = t1 - t0;
  std::chrono::milliseconds d = std::chrono::duration_cast< std::chrono::milliseconds >( fs );
  //----- Task 1 a -----//
  //Uncomment to activate helper code to retrieve N as commandline parameter in Task 1 c):
  int N;
  if (argc == 2)
  {
   N = std::stoi(argv[1]);
  } else 
  {
  std::cout << "Error: Missing problem size N. Please provide N as "
                "commandline parameter."
             << std::endl;
   exit(0);
  }

  double *v = new double[N];
  double *M = new double[N*N];

  for (int i = 0; i < N*N; i++){
    if (i < N){
      v[i] = i;
    }
    M[i] = i;
  }

  double *v_d, *M_d;
  cudaMalloc((void **)&v_d, sizeof(double) * N);
  cudaMalloc((void **)&M_d, sizeof(double) * N*N);
  cudaMemcpy(v_d, v, sizeof(double) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(M_d, M, sizeof(double) * N*N, cudaMemcpyHostToDevice);

  dim3 block(32, 32); //256 block size (hard coded, we actually want this to be at the max ideally )
  dim3 grid((N + block.x - 1) / block.x,
          (N + block.y - 1) / block.y);  //enough blocks to cover the grid
  t0 = std::chrono::high_resolution_clock::now();
  multi_vector_addition<<<grid,block>>>(N,v_d,M_d);
  cudaDeviceSynchronize();
  t1 =std::chrono::high_resolution_clock::now();
  cudaMemcpy(M,M_d,sizeof(double)*N*N,cudaMemcpyDeviceToHost);
  std::cout << "1a Results: \n";
  // for (int i =0; i< N*N;i++){
  //   std::cout << M[i] << "\n";
  // }
  fs = t1 - t0;
  std::cout<< fs.count() << " (s)\n";
  std::cout<< "\n";

  // delete[] v;
  // delete[] M;
  // cudaFree(v_d);
  // cudaFree(M_d);

  //task 1b

   for (int i = 0; i < N*N; i++){
    if (i < N){
      v[i] = i;
    }
    M[i] = i;
  }
  cudaMemcpy(v_d, v, sizeof(double) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(M_d, M, sizeof(double) * N*N, cudaMemcpyHostToDevice); //resetting the vectors/matrices
  t0 = std::chrono::high_resolution_clock::now();
  multi_vector_addition_shmem<<<grid,block>>>(N,v_d,M_d);
  cudaDeviceSynchronize();
  t1 = std::chrono::high_resolution_clock::now();
  cudaMemcpy(M,M_d,sizeof(double)*N*N,cudaMemcpyDeviceToHost);
  std::cout << "1b Results : \n";
  // for (int i =0; i< N*N;i++){
  //   std::cout << M[i] << "\n";
  // }
   fs = t1 - t0;
  std::cout<< fs.count() << " (s)\n";
  std::cout<< "\n";

  //task 1c
  for (int i = 0; i < N*N; i++){
    if (i < N){
      v[i] = i;
    }
    M[i] = i;
  }
  cudaMemcpy(v_d, v, sizeof(double) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(M_d, M, sizeof(double) * N*N, cudaMemcpyHostToDevice); //resetting the vectors/matrices


  //my intution actually says that in order to maximise the shared memory util we need tall, skinny blocks, and ideally with x*y = 1024
  //but my initial testing revealed that this was about half as fast as just using square blocks
int max_threads = 1024;
int b_y;

// if (N >= 512)
//     b_y = 512;  // tall blocks
// else if (N >= 128)
//     b_y = 128;
// else if (N >= 64)
//     b_y = 64;
// else
  b_y = 32;   // for very small N, just use N


int b_x = max_threads / b_y;

  dim3 block_dynam(b_x, b_y);
  dim3 grid_dynam((N + block_dynam.x - 1) / block_dynam.x,
          (N + block_dynam.y - 1) / block_dynam.y);  //enough blocks to cover the grid
  t0 = std::chrono::high_resolution_clock::now();
  multi_vector_addition_dynamic_shmem<<<grid_dynam,block_dynam,b_x*sizeof(double)>>>(N,v_d,M_d);
  cudaDeviceSynchronize();
  t1 = std::chrono::high_resolution_clock::now();
  cudaMemcpy(M,M_d,sizeof(double)*N*N,cudaMemcpyDeviceToHost);
  std::cout << "1c Results : \n";
  // for (int i =0; i< N*N;i++){
  //   std::cout << M[i] << "\n";
  // }
   fs = t1 - t0;
  std::cout<< fs.count() << " (s)\n";
  std::cout<< "\n";


  delete[] v;
  delete[] M;
  cudaFree(v_d);
  cudaFree(M_d);

  //----- Task 2 -----//
  int w = f_a(1);
  int x = f_b(2);
  int y = f_c(3);
  int z = f_d(w,x,y);
  std::cout << "Task 3 z = "<< z << "\n";

  //----- Task 4 -----//

  return EXIT_SUCCESS;

}
