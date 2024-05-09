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

#include<stdio.h>
#include<stdlib.h>
#include <iostream>
#include <vector>
#include <random>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#include "chypersphere.h"

//==================================================================== 80 ====>>

__global__ void compute_distances(int max_samples, 
                                    int max_dimensions,
                                    double* random_numbers,
                                    double* distances) {

   for (int i = 0; i < 1; ++i) {
      for (int j = 0; j < max_dimensions; ++j) {
         printf("%f => %f\n", random_numbers[(max_dimensions * i) + j], random_numbers[(max_dimensions * i) + j] * random_numbers[(max_dimensions * i) + j]);
         distances[i] += (random_numbers[(max_dimensions * i) + j] * random_numbers[(max_dimensions * i) + j]);
      }
   } 
}

void compute(const uint8_t min_dimensions, const uint8_t max_dimensions,
               const size_t max_samples) {
   
   // generate random numbers using std::random_device
   // https://stackoverflow.com/questions/1340729
   std::random_device rd;
   std::mt19937 e2(rd());
   std::uniform_real_distribution<double> distribution(0.0, 1.0);

   double* random_numbers;
   double* random_numbers_d;
   
   double * distances;
   double * distances_d;

   // https://stackoverflow.com/questions/9373929/cuda-transfer-2d-array-from-host-to-device
   random_numbers = (double*) malloc( max_dimensions * max_samples * sizeof(double));
   cudaMalloc(&random_numbers_d, max_dimensions * max_samples * sizeof(double));
   
   distances = (double*) malloc(max_samples * sizeof(double));
   cudaMalloc(&distances_d, max_samples * sizeof(double));
   
   for (size_t sample = 0; sample < max_samples; ++sample) {
      for (size_t dim = 0; dim < max_dimensions; ++dim) {
         random_numbers[(max_dimensions * sample) + dim] = distribution(e2);
         std::cout << random_numbers[(max_dimensions * sample) + dim] << std::endl << std::endl;

      }
   }
   
   cudaMemcpy(random_numbers_d, random_numbers, max_dimensions * max_samples * sizeof(double), cudaMemcpyHostToDevice);
   compute_distances<<<1, 1>>>(max_samples, max_dimensions, random_numbers_d, distances_d);
   cudaDeviceSynchronize();
   cudaMemcpy(distances, distances_d, max_samples*sizeof(double), cudaMemcpyDeviceToHost);

   for (size_t i = 0; i < 1; ++i) {
      // for (size_t j = 0; j < max_dimensions; ++j) {
         // std::cout << random_numbers[(max_dimensions * i) + j] << std::endl;
      // }
      std::cout << distances[i] << std::endl;
      std::cout << sqrt(distances[i]) << std::endl;
   }

   cudaFree(distances_d);
   cudaFree(random_numbers_d);
   free(random_numbers);
   free(distances);
}


int main() {
   compute(2, 16, 10'000);
   return 0;
}
//==================================================================== 80 ====>>