#include <algorithm>
#include <iostream>
#include <cmath>

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

	double s = function_a(u, v, N);
	double *x = function_b(u, v, N);
	double *y = function_c(A, x, N);
	double *w = function_d(A, x, u, N);
	double *z = function_e(s, w, y, N);

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
