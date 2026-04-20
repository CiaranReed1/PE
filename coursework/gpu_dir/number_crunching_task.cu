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
	auto t0 = std::chrono::high_resolution_clock::now();
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

	double *A;
	double *x;
	double *y;
	double *w;
	double *z;

	size_t NN = static_cast<size_t>(N) * static_cast<size_t>(N); //using this "pinned" host memory can enable more asynchronous memory transfers
	cudaMallocHost((void**)&A, sizeof(double) * NN);
	cudaMallocHost((void**)&x, sizeof(double) * N);
	cudaMallocHost((void**)&y, sizeof(double) * N);
	cudaMallocHost((void**)&w, sizeof(double) * N);
	cudaMallocHost((void**)&z, sizeof(double) * N);


	init_datastructures(u, v, A, N);

	//allocate device memory and copy data to device for u,v,x,w,y,A
	double *u_d;
	double *v_d;
	double *x_d;
	double *w_d;
	double *y_d;
	double *A_d;
	double *z_d;
	cudaMalloc((void **)&u_d, sizeof(double) * N);
	cudaMalloc((void **)&v_d,sizeof(double)*N);
	cudaMalloc((void **)&x_d,sizeof(double)*N);
	cudaMalloc((void **)&w_d,sizeof(double)*N);
	cudaMalloc((void **)&y_d,sizeof(double)*N);
	cudaMalloc((void **)&z_d,sizeof(double)*N);
	cudaMalloc((void **)&A_d,sizeof(double)*NN);
	cudaMemcpy(u_d, u, sizeof(double) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(v_d, v, sizeof(double) * N, cudaMemcpyHostToDevice);


	cudaStream_t streamMem, streamB, streamC, streamD;  //create streams
	cudaStreamCreate(&streamMem);
	cudaStreamCreate(&streamB);
	cudaStreamCreate(&streamC);
	cudaStreamCreate(&streamD);

	cudaEvent_t b_done, A_transferred, c_done; //create event to signal when b is done
	cudaEventCreate(&b_done);
	cudaEventCreate(&A_transferred);
	cudaEventCreate(&c_done);

	dim3 numBlocks(2*SM_count); //launch kernel b
	dim3 threadsPerBlock_b(256);  
	gpu_function_b<<<numBlocks, threadsPerBlock_b, 0, streamB>>>(N,u_d, v_d, x_d);
	cudaEventRecord(b_done,streamB);

	cudaMemcpyAsync(A_d,A,sizeof(double)*NN,cudaMemcpyHostToDevice,streamMem);  //asynchronously transfer A to GPU
	cudaEventRecord(A_transferred,streamMem);

	cudaStreamWaitEvent(streamC, b_done, 0); //makes streams C and D wait until b is done before starting (so x_d is available on the GPU)
	cudaStreamWaitEvent(streamD, b_done, 0); 
	cudaStreamWaitEvent(streamC, A_transferred, 0); //makes stream C and D wait until A is transferred before starting (so A_d is available on the GPU)
	cudaStreamWaitEvent(streamD, A_transferred, 0);

	cudaStreamWaitEvent(streamMem, b_done,0); //makes memory stream wait until b is done to transfer x back to host
	cudaMemcpyAsync(x,x_d,sizeof(double)*N,cudaMemcpyDeviceToHost,streamMem);

	int nthreads_c = 256; //launches kernels c and d 
	dim3 threadsPerBlock_c(nthreads_c);
	int nthreads_d = 256;
	dim3 threadsPerBlock_d(nthreads_d);
	gpu_function_c<<<numBlocks,threadsPerBlock_c,nthreads_c*sizeof(double),streamC>>>(N,A_d,x_d,y_d); 
	gpu_function_d<<<numBlocks,threadsPerBlock_d,nthreads_d*sizeof(double),streamD>>>(N,A_d,x_d,u_d,w_d);
	cudaEventRecord(c_done,streamC);

	cudaMemcpyAsync(w,w_d,sizeof(double)*N,cudaMemcpyDeviceToHost,streamD); //transfer w back to host when d is done

	cudaStreamWaitEvent(streamMem,c_done,0); //when c is done, transfer y back to host
	cudaMemcpyAsync(y,y_d,sizeof(double)*N,cudaMemcpyDeviceToHost,streamMem); 

	s = function_a(u, v, N); //run f_a concurrently on the CPU alongisde f_b, f_c and f_d on the GPU

	int nthreads_e = 256;
	dim3 threadsPerBlock_e(nthreads_e);
	gpu_function_e<<<numBlocks, threadsPerBlock_e,0,streamC>>>(N, s, x_d, y_d,z_d); 
	cudaMemcpyAsync(z, z_d, sizeof(double)*N, cudaMemcpyDeviceToHost, streamC);

	cudaDeviceSynchronize(); //global sync
	cudaStreamDestroy(streamMem); //destroy streams and events
	cudaStreamDestroy(streamB);
	cudaStreamDestroy(streamC);
	cudaStreamDestroy(streamD);
	cudaEventDestroy(b_done);
	cudaEventDestroy(A_transferred);
	cudaEventDestroy(c_done);

	cudaFree(A_d);
	cudaFree(u_d);
	cudaFree(v_d); 
	cudaFree(x_d);
	cudaFree(z_d); 
	cudaFree(y_d);
	cudaFree(w_d);

	print_results(s, x, y, z, A, w, N);

	delete[] u;
	delete[] v;
	cudaFreeHost(A);
	cudaFreeHost(x);
	cudaFreeHost(y);
	cudaFreeHost(w);
	cudaFreeHost(z);

	auto t1 = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
	std::ofstream file("cuda_tasks_timings.csv", std::ios::app);
	if (file.is_open()) {
	file << N << "," << duration << "\n";
	file.close();}

	return EXIT_SUCCESS;
}
