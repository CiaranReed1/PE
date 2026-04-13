CXX = g++
CXXFLAGS = -O3 -march=native

number_crunching: number_crunching.cpp
	module purge && module load gcc/14.2 && $(CXX) $(CXXFLAGS) -o number_crunching number_crunching.cpp

clean:
	rm number_crunching new_number_crunching
