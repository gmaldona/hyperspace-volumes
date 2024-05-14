# Volume of a D-dimensional Hypersphere
# see: https://cs.binghamton.edu/~kchiu/cs447/assign/final/
#
# Graduate student @ Thomas J. Watson College of Engineering and Applied
# Sciences, Binghamton University.
#
# author: Gregory Maldonado
# email : gmaldonado@cs.binghamton.edu
# date  : 2024-05-05
# web   : https://cs.binghamton.edu/~gmaldonado/
# file name:   hypersphere.hh
# description: Standard implementation of D-dimensional Hypersphere computation
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

CC     = g++
NVCC   = nvcc
CFLAGS = -std=c++17 -Wall -Wextra -pedantic -g # -O3
NVFLGS = -Xptxas -O3,-v
SRC    = src/impl
BUILD  = build/impl

all: simd cuda cpu

default: cpu # clion debugger can only target one dependency.

# SIMD Implementation <<=============================================== 80 ===>>
.PHONY: simd
simd: common.o simd.o
	$(CC) $(CLFAGS) -o build/ball_samp-simd \
	 			$(BUILD)/simd/shypersphere.o build/common/common.o

simd.o:
	mkdir -p $(BUILD)/simd
	$(CC) $(CFLAGS) -c $(SRC)/simd/shypersphere.cc \
						 -o $(BUILD)/simd/shypersphere.o

# CUDA Implementation <<=============================================== 80 ===>>

.PHONY: cuda
cuda: common.o cuda.o
	$(NVCC) $(CLFAGS) -o build/ball_samp-cuda \
				$(BUILD)/cuda/chypersphere.o build/common/common.o

cuda.o:
	mkdir -p $(BUILD)/cuda
	$(NVCC) $(NVFLGS) -c $(SRC)/cuda/chypersphere.cu \
						   -o $(BUILD)/cuda/chypersphere.o

# Standard Implementation <<=========================================== 80 ===>>

.PHONY: cpu
cpu: common.o cpu.o
	$(CC) $(CLFAGS) -o build/ball_samp-cpu \
				$(BUILD)/standard/hypersphere.o build/common/common.o

cpu.o:
	mkdir -p $(BUILD)/standard
	$(CC) $(CFLAGS) -c $(SRC)/standard/hypersphere.cc \
						 -o $(BUILD)/standard/hypersphere.o

# Common <<============================================================ 80 ===>>

common.o: .pre
	mkdir -p build/common
	$(CC) $(CFLAGS) -c src/common/common.cc -o build/common/common.o

.pre:
	mkdir -p build

.PHONY: clean
clean: 
	$(RM) -r build