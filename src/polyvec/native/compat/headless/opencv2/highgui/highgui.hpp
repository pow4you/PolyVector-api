// Headless replacement for OpenCV's highgui, used by the Python extension build.
//
// Upstream includes highgui everywhere but only calls imshow/waitKey in one
// debugging routine; these no-ops let the vendored sources compile and link
// without the highgui module (and its GUI toolkit dependencies).
#ifndef POLYVEC_HEADLESS_HIGHGUI_HPP
#define POLYVEC_HEADLESS_HIGHGUI_HPP

#include <string>
#include "opencv2/core.hpp"
// The real highgui.hpp includes these; upstream's main() relies on it for imread.
#include "opencv2/imgcodecs.hpp"

namespace cv {
inline void imshow(const std::string&, InputArray) {}
inline int waitKey(int = 0) { return -1; }
inline void namedWindow(const std::string&, int = 0) {}
inline void destroyAllWindows() {}
}

#endif
