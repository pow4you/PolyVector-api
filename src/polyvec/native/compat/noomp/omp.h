// Serial fallback for compilers without OpenMP (e.g. Apple Clang).
//
// Upstream calls omp_get_max_threads()/omp_get_thread_num() directly; without
// OpenMP the `#pragma omp` lines are ignored and these keep the code compiling.
#ifndef POLYVEC_NOOMP_H
#define POLYVEC_NOOMP_H
#ifndef _OPENMP
inline int omp_get_max_threads() { return 1; }
inline int omp_get_thread_num() { return 0; }
inline int omp_get_num_threads() { return 1; }
#endif
#endif
