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
* file name:   hypersphere.hh
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

#include <algorithm>
#include <vector>
#include <iostream>
#include <iomanip>

//==================================================================== 80 ====>>

/**
 *
 */
class histogram {

 public:
   histogram(const int bins, const double intervals, const size_t samples);
   void insert(double sample) {
       ++this->_histogram[(size_t)std::ceil(sample * bins) - 1];
   }
   double getIntervals() const;
   std::vector<int> getHistogram();
   std::vector<double> getRelativeFractionsHistogram();

   ~histogram() {
       this->_histogram.clear();

   }

 private:
   const int bins;
   const double intervals;
   const size_t samples;
   std::vector<int> _histogram;
};

/**
 *
 * @param n
 * @param bins
 * @param intervals
 * @return
 */
std::vector<histogram> generate_histograms(const size_t n,
                                           const int bins,
                                           const double intervals,
                                           const size_t samples) {
   std::vector<histogram> histograms;
   for (size_t i = 0; i < n; ++i) {
      histograms.emplace_back(bins, intervals, samples);
   }
   return histograms;
}

std::vector<double> histogram::getRelativeFractionsHistogram() {
    std::vector<double> mapped(_histogram.size());
    std::transform(_histogram.begin(), _histogram.end(), mapped.begin(), [&](int x){
        return (double)x / (double)this->samples;
    });
    return mapped;
}

/**
 *
 * @param os
 * @param _histogram
 * @return
 */
std::ostream& operator<<(std::ostream& os, histogram& _histogram) {
    std::vector<double> fractional = _histogram.getRelativeFractionsHistogram();
   for (size_t i = 0; i < fractional.size(); ++i) {
      os << fractional[i] << std::fixed << std::setprecision(3);
      if (i != fractional.size() - 1) {
         os << ",";
      }
   }
   os << std::endl;
   return os;
}

/**
 *
 * @param min_dimensions
 * @param max_dimensions
 * @param max_samples
 */
std::vector<histogram> compute(const uint8_t min_dimensions = 2,
                               const uint8_t max_dimensions = 16,
                               const size_t  max_samples    = 100'000);

//==================================================================== 80 ====>>