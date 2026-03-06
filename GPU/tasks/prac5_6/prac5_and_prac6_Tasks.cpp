#include <chrono>
#include <iostream>
#include <omp.h>


//----- Task 1 -----//
// a) Write an OpenMP CPU programme that adds the elements of a vector with N
//    elements to each row of a matrix with NxN elements. For this purpose,
//    write a function multi_vector_addition_CPU(...) that takes a vector of N
//    doubles and a matrix of NxN doubles as input.
// b) Use OpenMP's target directive to write a function
//    multi_vector_addition_GPU(...) that parallelizes this functionality on GPUs.
//    Ensure that the workload is distributed among teams *and* among threads in a
//    team. 
// c) Instrument your code with calls to std::chrono to measure the
//    execution times of your functions. Which header do you need to include?
//    As an example:
//     auto t0 = std::chrono::high_resolution_clock::now();
//     my_function(...);
//     auto t1 = std::chrono::high_resolution_clock::now();
//     std::chrono::duration< double > duration = t1 - t0;
//     std::chrono::milliseconds ms_duration = std::chrono::duration_cast<
//     std::chrono::milliseconds >( duration ); std::cout << duration.count() <<
//     "s\n"; std::cout << ms_duration.count() << "ms\n";
// d) Which version of the function runs faster? What could be the reason for
//    this?

void multi_vector_addition_CPU(const int N, double *vector, double *matrix) {

  int i,j;
  #pragma omp parallel default(none) shared(matrix, vector,N) private(i,j)
  {
    int thread_id = omp_get_thread_num();
    #pragma omp critical
    {
    std::cout<< "Runing on CPU, hello from thread "<< thread_id << "\n";
    }
    #pragma omp for
    for(i = 0; i<N; i++){
      for(j = 0; j < N; j++)
      {
        matrix[i*N + j] += vector[j];
      }
    }
  }
}

void multi_vector_addition_GPU(const int N, double *vector, double *matrix) {
  int i,j;
  #pragma omp target teams default(none) shared(matrix, vector,N) private(i,j) map(to : vector[0:N]) map(tofrom : matrix[0:N*N])
  {
    if (omp_is_initial_device()){
      printf("Running on CPU\n");
    } else
    {
      int num_teams = omp_get_num_teams();
      int team_id = omp_get_team_num();
      printf("Hello from GPU, team %d out of %d\n",team_id,num_teams);
      #pragma omp distribute  
      for(i = 0; i<N; i++)
      {
        #pragma omp parallel 
        {
        int thread_id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();

    
        if (thread_id == 0){
        printf("Team %d has %d threads\n", team_id, num_threads);
        }
        
        //printf("Hello from thread %d, within team %d\n",thread_id,team_id);
        
        #pragma omp for
        for(j = 0; j < N; j++)
          {
            matrix[i*N + j] += vector[j];
          }
        }  
      }
      
    }
  }
}

//----- Task 2 -----//
//   Consider the subsequent functions f_a(...), f_b(...), f_c(...), f_d(...)
//   and the data dependencies between their parameters as evident in the
//   main(...) function. 
//   a) Enable the concurrent execution of the functions on
//      the CPU via the OpenMP task depend clause to comply to data dependencies.
//   b) Adjust your code such that the functions f_a, f_b and f_c can be
//      executed concurrently on the GPU. Make use of OpenMP's reduction clause to adjust the code in f_a_gpu,
//      f_b_gpu and f_c_gpu accordingly.

void f_a(const int N,double *a, double *res) {
  double acc = 0;
  for (int i = 0; i < N; i++) {
    acc += a[i];
  }
  *res = acc;
}

void f_b(const int N, double *b, double *res) {
  double acc = 0;
  for (int i = 0; i < N; i++) {
    acc += b[i];
  }
  *res = acc;
}

void f_c(const int N, double *c, double *res) {
  double acc = 0;
  for (int i = 0; i < N; i++) {
    acc += c[i];
  }
  *res = acc;
}

void f_d(double a, double b, double c, double *res) { *res = a + b + c; }

void f_a_gpu(const int N, double *a, double *res) {
  double acc = 0;
  for (int i = 0; i < N; i++) {
    acc += a[i];
  }
  *res = acc;
}

void f_b_gpu(const int N, double *b, double *res) {
  double acc = 0;
  for (int i = 0; i < N; i++) {
    acc += b[i];
  }
  *res = acc;
}

void f_c_gpu(const int N, double *c, double *res) {
  double acc = 0;
  for (int i = 0; i < N; i++) {
    acc += c[i];
  }
  *res = acc;
}

//----- Code Template -----//

int main(int argc, char **argv) {
  //----- Task 1 -----//
  // Uncomment to activate helper code to retrieve N as commandline parameter if
  // you so wishs:
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

  double *vector = new double[N];
  double *matrix = new double[N * N];

  for (int i = 0; i < N; i++) {
    vector[i] = 1;
  }

  for (int i = 0; i < N * N; i++) {
    matrix[i] = 1;
  }

  auto t0 = std::chrono::high_resolution_clock::now();
  multi_vector_addition_CPU(N,vector, matrix);
  auto t1 = std::chrono::high_resolution_clock::now();
  std::chrono::duration< double > duration = t1 - t0;
  

  std::cout << "OpenMP CPU result: \n";
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      std::cout << matrix[i * N + j] << " ";
    }
    std::cout << "\n";
  }
  std::cout << "On the CPU this took : ";
  std::cout << duration.count() << "s\n"; 

  t0 = std::chrono::high_resolution_clock::now();
  multi_vector_addition_GPU(N,vector, matrix);
  t1 = std::chrono::high_resolution_clock::now();

  duration = t1 - t0;

  std::cout << "OpenMP GPU result: \n";
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      std::cout << matrix[i * N + j] << " ";
    }
    std::cout << "\n";
  }
  std::cout << "On the GPU this took : ";
  std::cout << duration.count() << "s\n"; 
  //----- Task 2 -----//
  // a)
  double *a = new double[N];
  double *b = new double[N];
  double *c = new double[N];
  for (int i = 0; i < N; i++) {
    a[i] = 1;
    b[i] = 1;
    c[i] = 1;
  }
  double w, x, y, z;

  f_a(N,a, &w);
  f_b(N,b, &x);
  f_c(N,c, &y);
  f_d(w, x, y, &z);
  std::cout << "The value of z is " << z << "."
            << "\n";

  //----- Task 2 -----//
  // b)
  f_a_gpu(N,a, &w);
  f_b_gpu(N,b, &x);
  f_c_gpu(N,c, &y);
  f_d(w, x, y, &z);
  std::cout << "The value of z is " << z << "."
            << "\n";

  return EXIT_SUCCESS;
}
