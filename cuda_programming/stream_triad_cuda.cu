#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

const int block_size = 512;

__global__ void set_values(const int N, const int M, float *A, float *x)
{
  const int column = threadIdx.x + blockIdx.x * blockDim.x;
  if (column < M){
    x[column] = column;
    for (unsigned int row = 0; row < N; ++row)
      {
        A[row + column*M] = row + column * M;
      }
  }
}


__global__ void matrix_vector_multiplication(const int    N,
                              const int M,
                              const float *A,
                              const float *x,
                              float *      y)
{
  const int row = threadIdx.x + blockIdx.x * blockDim.x;
  y[row] = 0.f;
  for (int col = 0; col < N; ++col)
    y[row] += A[row + M * col] * x[col];
    
}

void print_vector(const int N, const int M, std::vector<float> x)
{
  
  for(int row = 0; row < N; ++row){
    for(int column = 0; column < M; ++column){
      std::cout << x[row + M * column] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
  std::cout << std::endl;
  
}


// Run the actual benchmark
void benchmark_triad(
                     const long long   repeat, unsigned long N,unsigned long M)
{
  float *A, *x, *y;
  // allocate memory on the device

  cudaMalloc(&A, N * M * sizeof(float));
  cudaMalloc(&x, N * sizeof(float));
  cudaMalloc(&y, N * sizeof(float));

  const unsigned int n_blocks = (N + block_size - 1) / block_size;


  set_values<<<n_blocks, block_size>>>(N, M, A, x);
  cudaError_t error_code = cudaPeekAtLastError();
      if (error_code != cudaSuccess)
        std::cout << "CUDA returned the error " << cudaGetErrorString(error_code) << std::endl;


  const unsigned int           n_tests = 20;
  const unsigned long long int n_repeat =
    repeat > 0 ? repeat : std::max(1UL, 10U / N);
  double best = 1e10, worst = 0, avg = 0;
  for (unsigned int t = 0; t < n_tests; ++t)
    {
      // type of t1: std::chrono::steady_clock::time_point
      const auto t1 = std::chrono::steady_clock::now();

      for (unsigned int rep = 0; rep < n_repeat; ++rep)
      {
        matrix_vector_multiplication<<<n_blocks, block_size>>>(N, M, A, x, y);
      }

      cudaDeviceSynchronize();
      // measure the time by taking the difference between the time point
      // before starting and now
      const double time =
        std::chrono::duration_cast<std::chrono::duration<double>>(
          std::chrono::steady_clock::now() - t1)
          .count();

      best  = std::min(best, time / n_repeat);
      worst = std::max(worst, time / n_repeat);
      avg += time / n_repeat;
    }

  /*
  std::vector<float> y_result(N);
  std::vector<float> x_result(N);
  std::vector<float> A_result(N*M);


  cudaMemcpy(y_result.data(), y, N * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(x_result.data(), x, N * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(A_result.data(), A, N * M * sizeof(float), cudaMemcpyDeviceToHost);

  print_vector(N, 1, x_result);
  print_vector(N, M, A_result);
  print_vector(N, 1, y_result);
  */
  
    
  cudaFree(A);
  cudaFree(x);
  cudaFree(y);

  std::cout << "STREAM triad of size " << std::setw(8) << N
            << " : min/avg/max: " << std::setw(11) << best << " "
            << std::setw(11) << avg / n_tests << " " << std::setw(11) << worst
            << " seconds or " << std::setw(8) << 1e-6 * N / best
            << " MUPD/s or " << std::setw(8)
            << (1e-9 * sizeof(float) * (N*M + N + M)) / best << " GB/s" << std::endl;
}


int main(int argc, char **argv)
{
  std::cout << " Task1: first part" << std::endl;

   long repeat = -1;
   for (int n = 100; n <= 10000; n += 300)
   {
    int m = n;
    benchmark_triad(repeat, n, m);
   }
  return 0;
}
