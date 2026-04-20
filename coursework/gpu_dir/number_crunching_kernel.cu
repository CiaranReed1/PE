#include <algorithm>
#include <iostream>
#include <cmath>
#include <chrono>
#include <fstream>

// device properties
int SM_count;
int max_thread_per_block;
int max_thread_per_SM;
int warp_size;


void getDeviceProperties()
{
  struct cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  SM_count = prop.multiProcessorCount;
  max_thread_per_block = prop.maxThreadsPerBlock;
  warp_size = prop.warpSize;
  max_thread_per_SM = prop.maxThreadsPerMultiProcessor;
}

void append_timings(const std::string& filename,
                    int N,
                    double ta, double tb, double tc,
                    double td, double te)
{ //helper function to append function wall times to a file
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
{ //this function is fundamentally serial. I have not implemented a parallel approach (see report)

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

__global__ void gpu_function_b(const int N, const double *vec1, const double *vec2, double *res)
{	//function b kernel, accepts any 1D grid with 1D blocks
	int stride = gridDim.x * blockDim.x;
	int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
	for(unsigned int i = global_idx; i < N;i+=stride)
	{
		res[i] = vec1[i] + vec2[i];
	}
}

double *function_b(const double *u, const double *v, const int N, std::chrono::duration< double > *t_b_kern)
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

	auto t0 = std::chrono::high_resolution_clock::now();
	gpu_function_b<<<numBlocks, threadsPerBlock>>>(N,u_d, v_d, x_d);
	cudaDeviceSynchronize();
	auto t1 = std::chrono::high_resolution_clock::now();
	*t_b_kern = t1-t0;

	//retrieve results and free device memory
	cudaMemcpy(x,x_d,sizeof(double)*N,cudaMemcpyDeviceToHost);
	cudaFree(u_d);
	cudaFree(v_d);
	cudaFree(x_d);
	return x;
}

__global__ void gpu_function_c(const int N, const double *matrix, const double *vec, double *res )
{
	//each block calculates a single row, with threads calculating a partial strided sum over the row before reducing to a final result within the block.
	//accepts any 1D grid with 1D blocks
	int block_stride = gridDim.x;
	int thread_stride = blockDim.x;
	extern __shared__ double partial_sums[]; //shared memory to store thread partial sums, dynamic is used to set num of threads at runtime (this is technicaly not necessary as static shared memory would suffice)
	for (unsigned int i = blockIdx.x;i < N;i+= block_stride) //each block covers an entire row
	{ 
		//calculate thread based partial sum
		partial_sums[threadIdx.x] = 0;
		for (unsigned int j = threadIdx.x; j < N;j+= thread_stride) // threads use strided loop to construct partial sums
		{
			partial_sums[threadIdx.x] += matrix[i*N + j];
		}
		__syncthreads(); //sync threads before combining partial sums

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

double *function_c(const double *A, const double *x, const int N, std::chrono::duration< double > *t_c_kern) {
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
	auto t0 = std::chrono::high_resolution_clock::now();
	gpu_function_c<<<numBlocks,threadsPerBlock,nthreads*sizeof(double)>>>(N,A_d,x_d,y_d); //launch kernel, allocating dynamic memory based on N threads
	cudaDeviceSynchronize();
	auto t1 = std::chrono::high_resolution_clock::now();
	*t_c_kern = t1-t0;
	//retrieve results and free device memory
	
	cudaMemcpy(y,y_d,sizeof(double)*N,cudaMemcpyDeviceToHost);
	cudaFree(A_d);
	cudaFree(x_d);
	cudaFree(y_d);
	return y;
}

__global__ void gpu_function_d(const int N, const double *matrix, const double *vec_x, const double *vec_u, double *res)
{
	//This function is very similar to c, doing a reduction over each row of a matrix.
	//again the u[i] can be used to scale at the end
	//over the row, you start at j =0, then go up in 2s, i.e j = 0, 2,4 up until N-1. 
	//can do a similar strided thread approach, starting at threadidx * 2, (0,2,4) and having the stride as 2*Nthreads, (therefore next would be 6,8,10)
	//again accepts any 1D grid with 1D blocks
	int block_stride = gridDim.x;
	int thread_stride = 2* blockDim.x;
	extern __shared__ double partial_sums[];
	for(int i = blockIdx.x; i < N;i+=block_stride)
	{
		partial_sums[threadIdx.x] = 0;
		for(int j = threadIdx.x *2;j < N-1; j+=thread_stride)
		{
			partial_sums[threadIdx.x] += matrix[i*N + j] * vec_x[j];
		}
		__syncthreads();
		// tree based reduction into thread 0(enables parallelism in reduction)
		for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) { 
   			if (threadIdx.x < s) {
        		partial_sums[threadIdx.x] += partial_sums[threadIdx.x + s];
    		}
			__syncthreads(); //sync after each level of the tree
		} 
		if (threadIdx.x == 0){ //save result using single thread
			res[i] = vec_u[i] * partial_sums[0]; //scale final result using u[i]
		}
		__syncthreads();  //sync before moving onto next row 
	}
}

double *function_d(const double *A, const double *x, const double *u,
									 const int N, std::chrono::duration< double > *t_d_kern) {
	double *w = new double[N];
	
	//setup device memory
	double *A_d;
	double *x_d;
	double *u_d;
	double *w_d;
	size_t NN = static_cast<size_t>(N) * static_cast<size_t>(N);
	cudaMalloc((void **)&A_d,sizeof(double)*NN);
	cudaMalloc((void**)&x_d,sizeof(double)*N);
	cudaMalloc((void**)&u_d,sizeof(double)*N);
	cudaMalloc((void**)&w_d,sizeof(double)*N);
	cudaMemcpy(A_d,A,sizeof(double)*NN,cudaMemcpyHostToDevice);
	cudaMemcpy(x_d,x,sizeof(double)*N,cudaMemcpyHostToDevice);
	cudaMemcpy(u_d,u,sizeof(double)*N,cudaMemcpyHostToDevice);

	//launch kernel
	dim3 numBlocks(2*SM_count);
	int nthreads = 256;
	dim3 threadsPerBlock(nthreads);
	auto t0 = std::chrono::high_resolution_clock::now();
	gpu_function_d<<<numBlocks,threadsPerBlock,nthreads*sizeof(double)>>>(N,A_d,x_d,u_d,w_d); //launch kernel, allocating dynamic memory based on N threads
	cudaDeviceSynchronize();
	auto t1 = std::chrono::high_resolution_clock::now();
	*t_d_kern = t1-t0;

	//retrieve results and free device memory
	
	cudaMemcpy(w,w_d,sizeof(double)*N,cudaMemcpyDeviceToHost);
	cudaFree(A_d);
	cudaFree(x_d);
	cudaFree(u_d);
	cudaFree(w_d);
	return w;
}

__global__ void gpu_function_e(const int N, const double sum, const double *vec_x, const double *vec_y, double *res)
{
	int stride = blockDim.x * gridDim.x;
	int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
	double scale = 0;
	for (int i = global_idx; i<N;i+=stride)
	{
		scale = ((i & 1) == 0) ? sum : 1.0; //introduce scale variable to remove branch within warps
		res[i] = scale * vec_x[i] + vec_y[i];
	}
}

double *function_e(const double s, const double *x, const double *y,
									 const int N,std::chrono::duration< double > *t_e_kern) {
	double *z = new double[N];
	
	//setup device memory
	double *x_d;
	double *y_d;
	double *z_d;
	cudaMalloc((void**)&x_d,sizeof(double)*N);
	cudaMalloc((void**)&y_d,sizeof(double)*N);
	cudaMalloc((void**)&z_d,sizeof(double)*N);
	cudaMemcpy(x_d,x,sizeof(double)*N,cudaMemcpyHostToDevice);
	cudaMemcpy(y_d,y,sizeof(double)*N,cudaMemcpyHostToDevice);
	
	//launch kernel
	dim3 numBlocks(2*SM_count);
	int nthreads = 256;
	dim3 threadsPerBlock(nthreads);
	auto t0 = std::chrono::high_resolution_clock::now();
	gpu_function_e<<<numBlocks,threadsPerBlock>>>(N,s,x_d,y_d,z_d); //launch kernel, allocating dynamic memory based on N threads
	cudaDeviceSynchronize();
	auto t1 = std::chrono::high_resolution_clock::now();
	*t_e_kern = t1-t0;

	//retrieve results and free device memory

	cudaMemcpy(z,z_d,sizeof(double)*N,cudaMemcpyDeviceToHost);
	cudaFree(x_d);
	cudaFree(y_d);
	cudaFree(z_d);
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

	std::chrono::duration< double > t_b_kern;
	std::chrono::duration< double > t_c_kern;
	std::chrono::duration< double > t_d_kern;
	std::chrono::duration< double > t_e_kern;


	auto t0 = std::chrono::high_resolution_clock::now();
	double s = function_a(u, v, N);
	auto t1 = std::chrono::high_resolution_clock::now();
  	std::chrono::duration< double > t_a = t1 - t0;

	t0 = std::chrono::high_resolution_clock::now();
	double *x = function_b(u, v, N,&t_b_kern);
	t1 = std::chrono::high_resolution_clock::now();
	std::chrono::duration< double > t_b = t1 - t0;
	
	t0 = std::chrono::high_resolution_clock::now();
	double *y = function_c(A, x, N,&t_c_kern);
	t1 = std::chrono::high_resolution_clock::now();
	std::chrono::duration< double > t_c = t1 - t0;

	t0 = std::chrono::high_resolution_clock::now();
	double *w = function_d(A, x, u, N,&t_d_kern);
	t1 = std::chrono::high_resolution_clock::now();
	std::chrono::duration< double > t_d = t1 - t0;

	t0 = std::chrono::high_resolution_clock::now();
	double *z = function_e(s, w, y, N,&t_e_kern);
	t1 = std::chrono::high_resolution_clock::now();
	std::chrono::duration< double > t_e = t1 - t0;

	append_timings("cuda_kernel_timings.csv",N,
		t_a.count(),
		t_b_kern.count(),
		t_c_kern.count(),
		t_d_kern.count(),
		t_e_kern.count());

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
