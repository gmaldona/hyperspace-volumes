/*
* Volume of a D-dimensional Hypersphere
* see: https://cs.binghamton.edu/~kchiu/cs447/assign/final/
* 
* Graduate student @ Thomas J. Watson College of Engineering and Applied
* Sciences, Binghamton University.
*
* author: Gregory Maldonado
* email : gmaldonado@cs.binghamton.edu
* date  : 2024-05-05
* web   : https://cs.binghamton.edu/~gmaldonado/
*`
* file name:   chypersphere.cu
* description: D-dimensional Hypersphere computation using CUDA
*
*  THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/

#include <chrono>
#include <math.h>
#include<stdio.h>
#include<stdlib.h>
#include <iostream>
#include <vector>
#include <random>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#include "chypersphere.h"

using namespace std;
using namespace std::chrono;

//==================================================================== 80 ====>>

__global__ void compute_distances(int max_samples, 
                                    int dims,
                                    double* random_numbers,
                                    double* distances) {

   for (int sample = 0; sample < max_samples; ++sample) {
      for (int dim = 0; dim < dims; ++ dim){                               
            distances[sample] += ((random_numbers[(dims * sample) + dim]) * (random_numbers[(dims * sample) + dim]));
      }
   }
}

histogram::histogram(const int    _bins,
                     const double _intervals,
                     const size_t _samples) : bins(_bins),
                                              intervals(_intervals),
                                              samples(_samples),
                                              _histogram(_bins, 0) {
    this->_histogram = std::vector<int>(bins);
}

double histogram::getIntervals() const {
   return this->intervals;
}

std::vector<int> histogram::getHistogram() {
   return this->_histogram;
}

std::vector<histogram> compute(const uint8_t min_dimensions, const uint8_t max_dimensions,
               const size_t max_samples) {
   
   auto dimensional_histogram = generate_histograms(max_dimensions-1, 100, 0.01, max_samples);
   // dim - distance                                                                   
   std::vector<std::vector<double>> dimensional_samples(max_dimensions-1);

   double* random_numbers;
   double* random_numbers_d;
   double * distances;
   double * distances_d;

   

   for (uint8_t dims = min_dimensions; dims <= max_dimensions; ++dims) {

      // generate random numbers using std::random_device
      // https://stackoverflow.com/questions/1340729
      std::random_device rd;
      std::minstd_rand eng(dims);
      std::uniform_real_distribution<double> distribution(0.0, 1.0);

       // https://stackoverflow.com/questions/9373929/cuda-transfer-2d-array-from-host-to-device
   random_numbers = (double*) malloc(dims * max_samples * sizeof(double));
   cudaMalloc(&random_numbers_d, dims * max_samples * sizeof(double));

   distances = (double*) malloc(max_samples * sizeof(double));
   cudaMalloc(&distances_d, max_samples * sizeof(double));
   
      for (size_t sample = 0; sample < max_samples; ++sample) {
         for (size_t dim = 0; dim < dims; ++dim) {
            random_numbers[(dims * sample) + dim] = distribution(eng);
         }
      }

      cudaMemcpy(random_numbers_d, random_numbers, dims * max_samples * sizeof(double), cudaMemcpyHostToDevice);
      compute_distances<<<1, 256>>>(max_samples, dims, random_numbers_d, distances_d);
      cudaDeviceSynchronize();
      cudaMemcpy(distances, distances_d, max_samples*sizeof(double), cudaMemcpyDeviceToHost);

      for (size_t i = 0; i < max_samples; ++i) {
         double dist = sqrt(distances[i]);
         if (dist <= 1.0) {
            dimensional_histogram[dims - min_dimensions].insert(dist);
         }
      }
   cudaFree(distances_d);
   cudaFree(random_numbers_d);
   free(random_numbers);
   free(distances);
   }

   return dimensional_histogram;
}


int main() {
   auto start    = std::chrono::high_resolution_clock::now();
   auto dimensional_histogram = compute(2, 16, 100'000);
   auto stop     = std::chrono::high_resolution_clock::now();
   auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

   int dim = 2;
   for (auto& histogram : dimensional_histogram) {
      std::cout << dim << "-Dimension: " << std::endl;
      std::cout << histogram << std::endl;
      ++dim;
   }
   std::cerr << "time: " << duration.count() << " ms" << std::endl;

   return EXIT_SUCCESS;
}
//==================================================================== 80 ====>>