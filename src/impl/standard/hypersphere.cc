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
*
* file name:   hypersphere.cc
* description: Standard implementation of D-dimensional Hypersphere computation
*
*  THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/

#include <random>

#include "hypersphere.hh"

//==================================================================== 80 ====>>

histogram::histogram(const int    _bins,
                     const double _intervals,
                     const size_t _samples) : bins(_bins),
                                              intervals(_intervals),
                                              samples(_samples),
                                              _histogram(_bins, 0) {}

void histogram::insert(double sample) {
   ++this->_histogram[(size_t)std::ceil(sample * bins) - 1];
}

double histogram::getIntervals() const {
   return this->intervals;
}

std::vector<int> histogram::getHistogram() {
   return this->_histogram;
}

std::vector<double> histogram::getRelativeFractionsHistogram() {
   std::vector<double> mapped(_histogram.size());
   std::transform(_histogram.begin(), _histogram.end(), mapped.begin(), [&](int x){
      return (double)x / (double)this->samples;
   });
   return mapped;
}

std::vector<histogram> compute(const uint8_t min_dimensions,
             const uint8_t max_dimensions,
             const size_t  max_samples) {

   // generate random numbers using std::random_device
   // https://stackoverflow.com/questions/1340729
   std::random_device rd;
   std::mt19937 e2(rd());
   std::uniform_real_distribution<double> distribution(0.0, 1.0);

   auto dimensional_histogram = generate_histograms(max_dimensions - min_dimensions + 1,
                                                                      100, 0.01, max_samples);
   // dimension - sample - point
   std::vector<std::vector<std::vector<double>>>
      dimensional_samples(max_dimensions - min_dimensions + 1);

   for (uint8_t dims = min_dimensions; dims <= max_dimensions; ++dims) {
      size_t samples = 0;
      while (samples < max_samples) {
         std::vector<double> point(dims);
         for (uint8_t dim = 1; dim <= dims; ++dim) {
            point[dim] = distribution(e2);
         }
         double distance = std::accumulate(point.begin(), point.end(), 0.0,
                                           [](const double x, const double y) {
                                              return (x * x) + (y * y);
                                           });
         if (distance <= 1.0) {
            dimensional_samples[dims - min_dimensions].push_back(point);
            dimensional_histogram[dims - min_dimensions].insert(1 - distance);
            ++samples;
         }
      }
   }
   return dimensional_histogram;
}

int main() {
   auto dimensional_histogram= compute();

   int dim = 2;
   for (auto& histogram : dimensional_histogram) {
      std::cout << dim << "-Dimension: " << std::endl;
      std::cout << histogram << std::endl;
      ++dim;
   }

   return EXIT_SUCCESS;
}
//==================================================================== 80 ====>>