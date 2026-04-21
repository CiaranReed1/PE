CXX = g++
CXXFLAGS = -O3 -march=native

NVCC = nvcc
NVCCFLAGS = -O3

number_crunching: number_crunching.cpp
	module purge && module load gcc/14.2 && $(CXX) $(CXXFLAGS) -o number_crunching number_crunching.cpp

loopgpu: number_crunching_loop.cu
	module purge && module load cuda/12.0 && $(NVCC) $(NVCCFLAGS) -o loopgpu number_crunching_loop.cu

clean:
	rm -f number_crunching loopgpu new_number_crunching

taskgpu: number_crunching_task.cu
	module purge && module load cuda/12.0 && $(NVCC) $(NVCCFLAGS) -o taskgpu number_crunching_task.cu