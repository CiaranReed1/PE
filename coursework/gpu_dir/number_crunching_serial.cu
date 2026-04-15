#include <algorithm>
#include <iostream>
#include <cmath>
#include <chrono>
#include <fstream>

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

double function_a(const double *u, const double *v, const int N) {
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

double *function_b(const double *u, const double *v, const int N) {
	double *x = new double[N];
	for (unsigned int i = 0; i < N; i++) {
      x[i] = u[i] + v[i];
	}
	return x;
}

double *function_c(const double *A, const double *x, const int N) {
	double *y = new double[N];
	for (unsigned int i = 0; i < N; i++) {
		y[i] = 0;
	}
	for (unsigned int i = 0; i < N; i++) {
		for (unsigned int j = 0; j < N; j++) {
			if (i!=j) {
				y[i] += A[i * N + j] * x[i];
			}
		}
	}
	return y;
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
	t1 = std::chrono::high_resolution_clock::now();
	std::chrono::duration< double > t_c = t1 - t0;

	t0 = std::chrono::high_resolution_clock::now();
	double *w = function_d(A, x, u, N);
	t1 = std::chrono::high_resolution_clock::now();
	std::chrono::duration< double > t_d = t1 - t0;

	t0 = std::chrono::high_resolution_clock::now();
	double *z = function_e(s, w, y, N);
	t1 = std::chrono::high_resolution_clock::now();
	std::chrono::duration< double > t_e = t1 - t0;


	append_timings("serial_timings.csv",
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
