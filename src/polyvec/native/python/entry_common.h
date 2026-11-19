// Shared glue for the in-process entry points (polyvec._native._field/_flow).
//
// The upstream programs are written as one-shot command-line tools: a file in,
// an SVG out, progress on stdout, results kept in globals.  Calling them
// repeatedly from Python needs four things, provided here:
//
//   * NumPy <-> OpenCV/Eigen conversion,
//   * a per-call capture of everything written to std::cout / std::cerr,
//   * a process-wide lock (upstream keeps its working state in globals, so two
//     calls must never overlap), taken with the GIL released,
//   * C++ exceptions turned into a Python RuntimeError(message, log).
#ifndef POLYVEC_ENTRY_COMMON_H
#define POLYVEC_ENTRY_COMMON_H

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>

#include <cmath>
#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <mutex>
#include <sstream>
#include <streambuf>
#include <string>
#include <tuple>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core.hpp>

namespace nb = nanobind;

namespace polyvec {

using GrayImage = nb::ndarray<const std::uint8_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using Polylines = std::vector<std::vector<Eigen::Vector2d>>;
using StrokeArray = nb::ndarray<nb::numpy, double, nb::ndim<2>>;

// A streambuf that appends to a string under a mutex: upstream prints from
// inside OpenMP loops in a few places.
class LockedStringBuf : public std::streambuf {
public:
    std::string text;

protected:
    int_type overflow(int_type ch) override {
        if (ch != traits_type::eof()) {
            std::lock_guard<std::mutex> lock(mutex_);
            text.push_back(static_cast<char>(ch));
        }
        return ch;
    }
    std::streamsize xsputn(const char* s, std::streamsize n) override {
        std::lock_guard<std::mutex> lock(mutex_);
        text.append(s, static_cast<size_t>(n));
        return n;
    }

private:
    std::mutex mutex_;
};

// Redirects std::cout and std::cerr into a string for its lifetime.
class LogCapture {
public:
    LogCapture() : cout_(std::cout.rdbuf(&buffer_)), cerr_(std::cerr.rdbuf(&buffer_)) {}
    ~LogCapture() {
        std::cout.flush();
        std::cout.rdbuf(cout_);
        std::cerr.rdbuf(cerr_);
    }
    LogCapture(const LogCapture&) = delete;
    LogCapture& operator=(const LogCapture&) = delete;
    const std::string& text() const { return buffer_.text; }

private:
    LockedStringBuf buffer_;
    std::streambuf* cout_;
    std::streambuf* cerr_;
};

// Copy of a NumPy image as an 8-bit single-channel cv::Mat (owns its data, so
// it stays valid after the GIL is released).
inline cv::Mat to_mat(const GrayImage& image) {
    if (image.shape(0) < 2 || image.shape(1) < 2)
        throw std::invalid_argument("image must be at least 2x2 pixels");
    cv::Mat view(static_cast<int>(image.shape(0)), static_cast<int>(image.shape(1)), CV_8UC1,
                 const_cast<std::uint8_t*>(image.data()));
    return view.clone();
}

// Upstream writes each polyline to SVG as (x + 0.5, y + 0.5); do the same so
// in-process and command-line results share one coordinate convention.
// Empty curves are skipped exactly as upstream's SVG writers skip them.
inline nb::list to_strokes(const Polylines& polys) {
    nb::list strokes;
    for (const auto& poly : polys) {
        if (poly.empty())
            continue;
        const size_t count = poly.size();
        double* data = new double[count * 2];
        for (size_t i = 0; i < count; ++i) {
            data[2 * i] = poly[i].x() + 0.5;
            data[2 * i + 1] = poly[i].y() + 0.5;
        }
        nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<double*>(p); });
        strokes.append(StrokeArray(data, {count, 2}, owner));
    }
    return strokes;
}

inline nb::str to_pystr(const std::string& text) {
    // Upstream output is ASCII in practice; never fail on stray bytes.
    PyObject* str = PyUnicode_DecodeUTF8(text.data(), static_cast<Py_ssize_t>(text.size()), "replace");
    if (!str)
        throw nb::python_error();
    return nb::steal<nb::str>(str);
}

inline std::mutex& global_lock() {
    static std::mutex lock;
    return lock;
}

// Runs `work` (which must not touch Python objects) with the GIL released,
// the global lock held and stdout captured.  Returns (strokes, log); on a C++
// exception raises RuntimeError(message, log).
inline nb::tuple run_captured(const std::function<Polylines()>& work) {
    Polylines result;
    std::string log, error;
    bool failed = false;
    {
        nb::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(global_lock());
        LogCapture capture;
        try {
            result = work();
        } catch (const std::exception& exc) {
            failed = true;
            error = exc.what();
        } catch (...) {
            failed = true;
            error = "unknown C++ exception";
        }
        log = capture.text();
    }
    if (failed) {
        nb::object args = nb::make_tuple(to_pystr(error), to_pystr(log));
        PyErr_SetObject(PyExc_RuntimeError, args.ptr());
        throw nb::python_error();
    }
    return nb::make_tuple(to_strokes(result), to_pystr(log));
}

}  // namespace polyvec

#endif
