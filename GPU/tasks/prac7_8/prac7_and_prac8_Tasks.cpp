#include<iostream>
#include<vector>
#include<sycl/sycl.hpp>

int main(int argc, char **argv) 
{
  unsigned long N;
  if (argc == 2)
  {
   N = std::stoi(argv[1]);
  } else 
  {
  std::cout << "Error: Missing problem size N. Please provide N as commandline parameter."
            << std::endl;
   exit(0);
  }

  //----- Task 1 -----//
  // Use SYCL to query 5 device properties of your choice.
  std::cout<<"Task 1 : \n";
  auto devices = sycl::device::get_devices();
  for (auto d : devices){
    std::cout 
    << "Name : " << d.get_info<sycl::info::device::name>() << "\n"
    << "Vendor : "<<d.get_info<sycl::info::device::vendor>() << "\n"
    << "Version : "<< d.get_info<sycl::info::device::version>() << "\n"
    << "Max Compute Units : "<<d.get_info<sycl::info::device::max_compute_units>() << "\n"
    <<"Global mem size : "<<d.get_info<sycl::info::device::global_mem_size>() <<"\n";
  }

  //----- Task 2 -----//
  // a) Write a SYCL programme that implements the element-wise addition of two vectors 
  //    by using the high-level version of the parallel_for kernel invocation API.
  // b) Write a SYCL programme that implements the element-wise addition of two vectors 
  //    by using the ND range version of the parallel_for kernel invocation API.
  // c) Execute both programmes once on a CPU, and once on a GPU, by constructing a SYCL queue 
  //    with the corresponding device selectors. 

  std::vector<double> x(N);
  std::vector<double> y(N);
  std::vector<double> z(N);

  for(auto& i : x) i = 1.0;
  for(auto& i : y) i = 1.0;
  for(auto& i : z) i = 0.0;

  // a)
  std::cout<<"\n Task 2a : \n";
  sycl::queue q;  
  {
    sycl::buffer<double,1> x_buf(x.data(),x.size());
    sycl::buffer<double,1> y_buf(y.data(),y.size());
    sycl::buffer<double,1> z_buf(z.data(),z.size());
    q.submit([&](sycl::handler& cgh)
    {
      auto x_d = x_buf.get_access<sycl::access::mode::read>(cgh);
      auto y_d = y_buf.get_access<sycl::access::mode::read>(cgh);
      auto z_d = z_buf.get_access<sycl::access::mode::write>(cgh);
      cgh.parallel_for(
        sycl::range<1>(N),[=](sycl::id<1> i)
        {
          z_d[i] = x_d[i] + y_d[i]; 
        }
      );
    });
  }

  std::cout << "Vector addition result (high lvl parallel for): \n";
  for(auto i : z) std::cout << i << " ";
  std::cout << "\n";

  // b)
  sycl::range global{2, 2, 2};
  sycl::range local{2, 2, 2}; 
  {
    sycl::buffer<double,1> x_buf(x.data(),x.size());
    sycl::buffer<double,1> y_buf(y.data(),y.size());
    sycl::buffer<double,1> z_buf(z.data(),z.size());
    q.submit([&](sycl::handler& cgh)
    {
      auto x_d = x_buf.get_access<sycl::access::mode::read>(cgh);
      auto y_d = y_buf.get_access<sycl::access::mode::read>(cgh);
      auto z_d = z_buf.get_access<sycl::access::mode::write>(cgh);
      cgh.parallel_for(
        sycl::nd_range{global,local},[=](sycl::nd_item<3> i)
        {
          int j = i.get_global_linear_id();
          z_d[j] = x_d[j] + y_d[j]; 
        }
      );
    });
  }

  std::cout <<"\n Task 2b \n";
  std::cout << "Vector addition result (nd range): \n";
  for(auto i : z) std::cout << i << " ";
  std::cout << "\n";

  //task 2c

  std::cout<< "\n Task 2c \n";
  sycl::queue q_cpu{sycl::host_selector{}};
  std::cout << "Running on CPU: "
            << q_cpu.get_device().get_info<sycl::info::device::name>()
            << "\n";

   {
    sycl::buffer<double,1> x_buf(x.data(),x.size());
    sycl::buffer<double,1> y_buf(y.data(),y.size());
    sycl::buffer<double,1> z_buf(z.data(),z.size());
    q_cpu.submit([&](sycl::handler& cgh)
    {
      auto x_d = x_buf.get_access<sycl::access::mode::read>(cgh);
      auto y_d = y_buf.get_access<sycl::access::mode::read>(cgh);
      auto z_d = z_buf.get_access<sycl::access::mode::write>(cgh);
      cgh.parallel_for(
        sycl::range<1>(N),[=](sycl::id<1> i)
        {
          z_d[i] = x_d[i] + y_d[i]; 
        }
      );
    }).wait();
  }
   std::cout << "Vector addition result (high lvl parallel for) (CPU): \n";
  for(auto i : z) std::cout << i << " ";
  std::cout << "\n";
  {
    sycl::buffer<double,1> x_buf(x.data(),x.size());
    sycl::buffer<double,1> y_buf(y.data(),y.size());
    sycl::buffer<double,1> z_buf(z.data(),z.size());
    q_cpu.submit([&](sycl::handler& cgh)
    {
      auto x_d = x_buf.get_access<sycl::access::mode::read>(cgh);
      auto y_d = y_buf.get_access<sycl::access::mode::read>(cgh);
      auto z_d = z_buf.get_access<sycl::access::mode::write>(cgh);
      cgh.parallel_for(
        sycl::nd_range{global,local},[=](sycl::nd_item<3> i)
        {
          int j = i.get_global_linear_id();
          z_d[j] = x_d[j] + y_d[j]; 
        }
      );
    }).wait();
  }
    std::cout << "Vector addition result (nd range) (CPU): \n";
  for(auto i : z) std::cout << i << " ";
  std::cout << "\n";

  sycl::queue q_gpu{sycl::gpu_selector{}};
std::cout << "Running on GPU: "
          << q_gpu.get_device().get_info<sycl::info::device::name>()
          << "\n";
   {
    sycl::buffer<double,1> x_buf(x.data(),x.size());
    sycl::buffer<double,1> y_buf(y.data(),y.size());
    sycl::buffer<double,1> z_buf(z.data(),z.size());
    q_gpu.submit([&](sycl::handler& cgh)
    {
      auto x_d = x_buf.get_access<sycl::access::mode::read>(cgh);
      auto y_d = y_buf.get_access<sycl::access::mode::read>(cgh);
      auto z_d = z_buf.get_access<sycl::access::mode::write>(cgh);
      cgh.parallel_for(
        sycl::range<1>(N),[=](sycl::id<1> i)
        {
          z_d[i] = x_d[i] + y_d[i]; 
        }
      );
    }).wait();
  }
   std::cout << "Vector addition result (high lvl parallel for) (GPU): \n";
  for(auto i : z) std::cout << i << " ";
  std::cout << "\n";
  {
    sycl::buffer<double,1> x_buf(x.data(),x.size());
    sycl::buffer<double,1> y_buf(y.data(),y.size());
    sycl::buffer<double,1> z_buf(z.data(),z.size());
    q_gpu.submit([&](sycl::handler& cgh)
    {
      auto x_d = x_buf.get_access<sycl::access::mode::read>(cgh);
      auto y_d = y_buf.get_access<sycl::access::mode::read>(cgh);
      auto z_d = z_buf.get_access<sycl::access::mode::write>(cgh);
      cgh.parallel_for(
        sycl::nd_range{global,local},[=](sycl::nd_item<3> i)
        {
          int j = i.get_global_linear_id();
          z_d[j] = x_d[j] + y_d[j]; 
        }
      );
    }).wait();
  }
    std::cout << "Vector addition result (nd range) (GPU): \n";
  for(auto i : z) std::cout << i << " ";
  std::cout << "\n";


  //----- Task 3 -----//
  // Write a SYCL programme that computes the accumulated sum of the elements of a vector.
  
  // Start out with a naive implementation with a global atomic variable.
  // For more performant solutions, see: 
  // https://www.intel.com/content/www/us/en/develop/documentation/oneapi-gpu-optimization-guide/top/kernels/reduction.html
  // Please note that clang++ does not support the built-in reduction as proposed by the Intel extensions.
  
  std::vector<double> u(N);
  for(auto& i : u) i = 1.0;
  double acc_u = 0;
  {
      sycl::buffer<double,1> u_buf(u.data(),u.size());
      sycl::buffer<double,1> acc_buf(&acc_u, 1);
      q.submit([&](sycl::handler& cgh){
       auto u_d = u_buf.get_access<sycl::access::mode::read>(cgh);
       auto acc_d = acc_buf.get_access<sycl::access::mode::read_write>(cgh);
       cgh.parallel_for(
        sycl::range<1>(N),[=](sycl::id<1> i)
        {
         sycl::atomic_ref<double, 
                             sycl::memory_order::relaxed, 
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space> 
                atomic_acc(acc_d[0]);
            atomic_acc.fetch_add(u_d[i]);
        }
      );
      }).wait();
  }
  
  std::cout << "Reduction result:" << "\n";
  std::cout << acc_u << "\n";

  //----- Task 4 -----//
  // Consider the following article on the execution order of SYCL kernels: 
  // https://developer.codeplay.com/products/computecpp/ce/2.11.0/guides/sycl-guide/multiple-kernels
  // Under the assumption that all of the kernel functions you have written in Tasks 1 to 3 are submitted
  // to the same queue, determine which of them may be executed concurrently.

  //My intuition is that given that they all write to the same z buffer, they will not be able to 

  std::cout << "\n Task 4 \n";
  std::cout << "My intuition is that given that they all write to the same host variable "z", they will not be able to run concurrently. That being said i redefined the buffers for each kernel, so maybe they will try to run concurrently?Even though they would overwrite the host variable z?\n";
  std::cout<<"Task 3 depends on completely different variables and buffers so i expect this to be able to run concurrently\n";
  
  return EXIT_SUCCESS;

}
