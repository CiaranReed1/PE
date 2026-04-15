#include <algorithm>
#include <iostream>
#include <cmath>
#include <chrono>
#include <fstream>

int SM_count;
int max_thread_per_block;
int max_thread_per_SM;
int warpSize;

void getDeviceProperties()
{
  struct cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  SM_count = prop.multiProcessorCount;
  max_thread_per_block = prop.maxThreadsPerBlock;
  warpSize = prop.warpSize;
  max_thread_per_SM = prop.maxThreadsPerMultiProcessor;
}

void append_timings(const std::string& filename,
                    int N,
                    double ta, double tb, double tc,
                    double td, double te)
{
    std::ofstream file(filename, std::ios::app);

    if (file.is_open()) {
        file << N << ","
             << ta << ","
             << tb << ","
             << tc << ","
             << td << ","
             << te << "\n";
        file.close();
    }
}

double function_a(const double *u, const double *v, const int N) 
{

	double s = 0;
	for (unsigned int i = 0; i < N; i++) 
	{
		if (s < 10)
		{
			s += u[i];
		} 
		else
		{
			s += u[i]*v[i];
		}
	}
	s /= sqrt((double)N);
	return s;
}


double *function_b(const double *u, const double *v, const int N)
{
	double *x = new double[N];

	//setup device memory
	double *u_d;
	double *v_d;
	double *x_d;
	cudaMalloc((void **)&u_d, sizeof(double) * N);
	cudaMalloc((void **)&v_d,sizeof(double)*N);
	cudaMalloc((void **)&x_d,sizeof(double)*N);
	cudaMemcpy(u_d, u, sizeof(double) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(v_d, v, sizeof(double) * N, cudaMemcpyHostToDevice);

	//launch kernel
	dim3 numBlocks(2*SM_count);

	dim3 threadsPerBlock(256);  
	gpu_function_b<<<numBlocks, threadsPerBlock>>>(N,u_d, v_d, x_d);
	cudaDeviceSynchronize();

	//retrieve results and free device memory
	cudaMemcpy(x,x_d,sizeof(double)*N,cudaMemcpyDeviceToHost);
	cudaFree(u_d);
	cudaFree(v_d);
	cudaFree(x_d);
	return x;
}

__global__ void gpu_function_b(const int N, const double *vec1, const double *vec2, double *res)
{	
	int stride = gridDim.x * blockDim.x;
	int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
	for(unsigned int i = global_idx; i < N;i+=stride)
	{
		res[i] = vec1[i] + vec2[i];
	}
}

double *function_c(const double *A, const double *x, const int N) {
	double *y = new double[N];
	
	//setup device memory
	double *A_d;
	double *x_d;
	double *y_d;
	size_t NN = static_cast<size_t>(N) * static_cast<size_t>(N);
	cudaMalloc((void **)&A_d,sizeof(double)*NN);
	cudaMalloc((void**)&x_d,sizeof(double)*N);
	cudaMalloc((void**)&y_d,sizeof(double)*N);
	cudaMemcpy(A_d,A,sizeof(double)*NN,cudaMemcpyHostToDevice);
	cudaMemcpy(x_d,x,sizeof(double)*N,cudaMemcpyHostToDevice);

	//launch kernel
	dim3 numBlocks(2*SM_count);
	int nthreads = 256;
	dim3 threadsPerBlock(nthreads);
	gpu_function_c<<<numBlocks,threadsPerBlock,nthreads*sizeof(double)>>>(N,A_d,x_d,y_d);

	//retrieve results and free device memory
	cudaDeviceSynchronize();
	cudaMemcpy(y,y_d,sizeof(double)*N,cudaMemcpyDeviceToHost);
	cudaFree(A_d);
	cudaFree(x_d);
	cudaFree(y_d);
	return y;
}

__global__ void gpu_function_c(const int N, const double *matrix, const double *vec, double *res )
{
	//each block calculates a signle row, with threads calculating a partial strided sum over the row before reducing to a final result within the block.
	//then a strided approach is used to ensure that the blocks cover every single row
	int stride = gridDim.x;
	extern __shared__ double partial_sums[]; //shared memory to store thread partial sums, dynamic is used to set num of threads at runtime dependent on N
	for (unsigned int i = blockIdx.x;i < N;i+= stride)
	{
		//calculate thread based partial sum
		partial_sums[threadIdx.x] = 0;
		for (unsigned int j = threadIdx.x; j < N;j+= blockDim.x) // strided approach to ensure entire row is calculated
		{
			partial_sums[threadIdx.x] += matrix[i*N + j];
		}
		__syncthreads(); //sync before combining partial sums

		// tree based reduction into thread 0(enables parallelism in reduction)
		for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) { 
   			if (threadIdx.x < s) {
        		partial_sums[threadIdx.x] += partial_sums[threadIdx.x + s];
    		}
			__syncthreads(); //sync after each level of the tree
		} 

		if (threadIdx.x == 0){ //save result using single thread
			partial_sums[0] -= matrix[i*N + i]; //remove diagonal element
			res[i] = partial_sums[0]*vec[i]; //scale by vector
		}
		__syncthreads();  //sync before moving onto next row 
	}
}

double *function_d(const double *A, const double *x, const double *u,
									 const int N) {
	double *w = new double[N];
	for (unsigned int i = 0; i < N; i++) {
		w[i] = 0.0;
		for (unsigned int j = 0; j < N-1; j+=2) {
			w[i] += u[i] * A[i * N + j] * x[j];
		}
	}
	return w;
}

double *function_e(const double s, const double *x, const double *y,
									 const int N) {
	double *z = new double[N];
	for (unsigned int i = 0; i < N; i++) {
		if (i % 2 == 0) {
			z[i] = s * x[i] + y[i];
		} else {
			z[i] = x[i] + y[i];
		}
	}
	return z;
}

void init_datastructures(double *u, double *v, double *A, const int N) {
	for (unsigned int i = 0; i < N; i++) {
		u[i] = 0.1;
		v[i] = 1.0 - u[i];
	}

	for (unsigned int i = 0; i < N * N; i++) {
		A[i] = u[i % N] * v[i / N];
	}
}

void print_results(const double s, const double *x, const double *y,
									 const double *z, const double *A, const double *w,
									 const int N) {
	std::cout << "s: " << std::endl << s << std::endl;
	int M = std::min(N, 16);
	std::cout << "x: " << std::endl;
	for (unsigned int i = 0; i < M; i++) {
		std::cout << x[i] << " ";
	}
	std::cout << std::endl;

	std::cout << "y: " << std::endl;
	for (unsigned int i = 0; i < M; i++) {
		std::cout << y[i] << " ";
	}
	std::cout << std::endl;

	std::cout << "z: " << std::endl;
	for (unsigned int i = 0; i < M; i++) {
		std::cout << z[i] << " ";
	}
	std::cout << std::endl;

	std::cout << "w: " << std::endl;
	for (unsigned int i = 0; i < M; i++) {
		std::cout << w[i] << " ";
	}
	std::cout << std::endl;

	std::cout << "A: " << std::endl;
	for (unsigned int i = 0; i < M; i++) {
		for (unsigned int j = 0; j < M; j++) {
			std::cout << A[i * N + j] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

int main(int argc, char **argv) {
	getDeviceProperties();
	int N;

	if (argc == 2) {
		N = std::stoi(argv[1]);
	} else {
		std::cout << "Error: Missing problem size N. Please provide N as "
								 "commandline parameter."
							<< std::endl;
		exit(0);
	}

	double *u = new double[N];
	double *v = new double[N];
	double *A = new double[N * N];

	init_datastructures(u, v, A, N);

	auto t0 = std::chrono::high_resolution_clock::now();
	double s = function_a(u, v, N);
	auto t1 = std::chrono::high_resolution_clock::now();
  	std::chrono::duration< double > t_a = t1 - t0;

	t0 = std::chrono::high_resolution_clock::now();
	double *x = function_b(u, v, N);
	t1 = std::chrono::high_resolution_clock::now();
	std::chrono::duration< double > t_b = t1 - t0;
	
	t0 = std::chrono::high_resolution_clock::now();
	double *y = function_c(A, x, N);
	cudaDeviceSynchronize();
	t1 = std::chrono::high_resolution_clock::now();
	std::chrono::duration< double > t_c = t1 - t0;

	t0 = std::chrono::high_resolution_clock::now();
	double *w = function_d(A, x, u, N);
	cudaDeviceSynchronize();
	t1 = std::chrono::high_resolution_clock::now();
	std::chrono::duration< double > t_d = t1 - t0;

	t0 = std::chrono::high_resolution_clock::now();
	double *z = function_e(s, w, y, N);
	cudaDeviceSynchronize();
	t1 = std::chrono::high_resolution_clock::now();
	std::chrono::duration< double > t_e = t1 - t0;


	append_timings("cuda_timings.csv",
               N,
               t_a.count(),
               t_b.count(),
               t_c.count(),
               t_d.count(),
               t_e.count());
			
	print_results(s, x, y, z, A, w, N);

	delete[] u;
	delete[] v;
	delete[] w;
	delete[] A;
	delete[] x;
	delete[] y;
	delete[] z;

	return EXIT_SUCCESS;
}
