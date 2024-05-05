# Volume of a D-dimensional Hypersphere

CS 547 High Performance Computing @ Binghamton University
* release date 2024-04-29 (piazza)

## Submission

* Deadline:
   * 2024-05-13 11:59 PM

Last working commit:
* [Insert last working commit]

## Execution
Included within the repository is a [Makefile](Makefile).

### CPU (Standard) Implementation
```shell
make cpu && ./build/ball_samp-cpu
```

### CUDA Implmentation
```shell
make cuda && ./build/ball_samp-cuda
```

### SIMD Implementation
```shell
make simd && ./build/ball_samp-simd
```

## Source
* https://cs.binghamton.edu/~kchiu/cs447/assign/final/