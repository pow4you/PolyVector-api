// Force-included into every upstream translation unit of the Python extension.
//
// The upstream programs report progress on stdout, mostly through std::cout
// (captured per call by polyvec::LogCapture, see entry_common.h) and in a few
// places through printf.  Routing printf to std::cout keeps that output out of
// the host process's stdout too, so a library call stays silent and the whole
// log is returned to Python instead.
#ifndef POLYVEC_LOG_H
#define POLYVEC_LOG_H

#include <cstdarg>
#include <cstdio>
#include <iostream>

inline int polyvec_log_printf(const char* format, ...) {
    char buffer[2048];
    va_list args;
    va_start(args, format);
    const int length = std::vsnprintf(buffer, sizeof buffer, format, args);
    va_end(args);
    std::cout << buffer;
    return length;
}

// Not in the module's own file (python/*_module.cpp), which also includes the
// Python and nanobind headers.
#ifndef POLYVEC_ENTRY_TU
#define printf(...) polyvec_log_printf(__VA_ARGS__)
#endif

#endif
