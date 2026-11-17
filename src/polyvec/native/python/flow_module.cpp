// polyvec._native._flow: PolyVector Flow (Puhachov et al. 2021) in-process.
//
// The vendored program (../../vendor/pvflow/src) is compiled unchanged.  Its
// main.cpp is included below with `main` renamed, which brings in upstream's
// globals and helper routines.  vectorize_flow() runs the same pipeline as
// that main(), step for step, except that
//
//   * the image and keypoints come from NumPy arrays instead of FILE.png and
//     FILE.pts,
//   * the selected curves are returned instead of being written to OUT.svg,
//   * calls that only feed the Qt debug viewer (mw.*), and the per-iteration
//     copies of every curve kept solely for that viewer, are left out.
//
// The program solves its curve-selection problem through the Gurobi API; here,
// as in the Docker image, that API is provided by compat/gurobi on top of HiGHS.
//
// Every global main() relies on is reassigned at the start of each call, so
// repeated calls are independent.

#include "entry_common.h"

// main() refers to a QApplication named `a` (only to call a.exec()).
#include <QtCore/QCoreApplication>  // the headless stand-in (compat/headless)
static PolyvecHeadlessApp a;

#define main polyvec_upstream_flow_main
#include "main.cpp"
#undef main

namespace {

struct FlowInputs {
	cv::Mat gray;
	std::vector<cv::Point2d> pts;   // upstream order: (row, column)
	std::vector<int> pts_types; // 1 endpoint, 3 junction, 5 sharp corner
	int flowIterations = 1500;
};

polyvec::Polylines vectorize_flow(const FlowInputs& in)
{
	using namespace cv;

	std::vector<cv::Point2d> pts = in.pts;
	std::vector<int> pts_types = in.pts_types;
	int flowIterations = in.flowIterations;

	// main(): imread(FILE, IMREAD_COLOR) + cvtColor(BGR2GRAY) == the gray image.
	bwImg = Scalar(255) - in.gray;
	m = bwImg.rows; n = bwImg.cols; //m: height, n: width

	//fill in the mask
	cv::Mat actualMask;
	threshold(bwImg, actualMask, BACKGROUND_FOREGROUND_THRESHOLD, 255, THRESH_BINARY);

	for (int i = 0; i < 3; ++i)
		repairMask(actualMask);

	origMask = actualMask;

	calculateGradient();
	calculateWeight();
	int nnz = 0;
	auto indices = calculateIndices(origMask, nnz);

	std::cout << "Optimizing...";
	Eigen::VectorXcd X  = optimizeByLinearSolve(bwImg, weight, tau, FRAME_FIELD_SMOOTHNESS_WEIGHT, origMask, indices);

	std::cout << "Finding roots.. ";
	auto roots = findRoots(X, origMask);

	auto singularities = findSingularities(roots, X, indices, origMask);
	bool improved;
	int totalNSingularities = 0;
	do
	{
		int origSingularityCount = singularities.size();

		bool somethingNew = false;
		for (auto s : singularities)
		{
			if (weight(s[0], s[1]) > 1e-5)
			{
				somethingNew = true;
				weight(s[0], s[1]) = 0;
				totalNSingularities++;
			}
		}
		if (!somethingNew)
			break;

		X = optimizeByLinearSolve(bwImg, weight, tau, FRAME_FIELD_SMOOTHNESS_WEIGHT, origMask, indices);
		roots = findRoots(X, origMask);
		singularities = findSingularities(roots, X, indices, origMask);

		std::cout << "done (" << origSingularityCount - (int)singularities.size() << " singularities removed" << std::endl;
		improved = origSingularityCount - singularities.size() > 0;
	} while (improved);

	//now trace
	std::cout << "Tracing... ";
	std::map<std::array<int, 2>, std::vector<PixelInfo>> pixelInfo;
	std::vector<std::array<bool, 2>> endedWithASingularity;
	auto polys = traceAll(bwImg, actualMask, actualMask, roots, X, indices, pixelInfo, endedWithASingularity);

	auto singularitiesWithMask = findSingularities(roots, X, indices, origMask);
	auto reebGraph = computeAlmostReebGraph(origMask, roots, polys, pixelInfo, singularities, indices, X, endedWithASingularity);
	findWidths(reebGraph);

	contractSingularityBranches(reebGraph);
	simpleThresholds(reebGraph);
	edge_iter eit, eend;
	for (std::tie(eit, eend) = boost::edges(reebGraph); eit != eend; ++eit)
	{
		reebGraph[*eit].weight = 1;
	}
	contractLoops(reebGraph, origMask, polys);

	// (main() reads the keypoints from FILE.pts here; they arrive in `in`.)

	//make sure the edge weights are correct
	for (auto [eit, eend] = boost::edges(reebGraph); eit != eend; ++eit)
	{
		reebGraph[*eit].weight = (reebGraph[eit->m_target].location - reebGraph[eit->m_source].location).norm()+1e-9;
	}

	std::map<vertex_descriptor, bool> covered;

	//extend the frame field to the rest of the image
	cv::Mat fullImageMask = cv::Mat::ones(m, n, CV_8U);
	const double dilation_size = 5.0;
	cv::Mat element = getStructuringElement(cv::MORPH_ELLIPSE,
		Size(2 * dilation_size + 1, 2 * dilation_size + 1),
		Point(dilation_size, dilation_size));
	dilate(origMask, fullImageMask, element);

	int newNnz;
	auto newIndices = calculateIndices(fullImageMask, newNnz);
	auto Xextended = optimizeByLinearSolve_holdingSomeFixed(bwImg, weight, tau, FRAME_FIELD_SMOOTHNESS_WEIGHT, fullImageMask, origMask, newIndices, indices, X);

	Eigen::MatrixXcd c0(m, n), c2(m, n);

	int idx = 0;
	for (int j = 0; j < n; ++j) {
		for (int i = 0; i < m; ++i) {
			if (fullImageMask.at<uchar>(i, j) != 0) {
				c0(i, j) = Xextended(idx);
				c2(i, j) = Xextended(idx + newNnz);
				idx++;
			} else {
				c0(i, j) = 0;
				c2(i, j) = 0;
			}
		}
	}

	// set to true if you want to scale the framefield outside the original mask
	bool scale_ff = true;
	if(scale_ff)
	{
		Mat inverse_origMask;
		Mat dist;
		cv::bitwise_not(origMask, inverse_origMask);
		cv::distanceTransform(inverse_origMask,dist,DIST_L1,3,CV_8U);
		for (int j = 0; j < n; ++j) {
			for (int i = 0; i < m; ++i) {
				if ((fullImageMask.at<uchar>(i, j) != 0) && (origMask.at<uchar>(i, j) == 0)) {
					c0(i, j) = c0(i, j) * 2.0 * static_cast<double>(dist.at<uchar>(i, j));
					c2(i, j) = c2(i, j) * 2.0 * static_cast<double>(dist.at<uchar>(i, j));
				}
			}
		}
	}

	FrameFieldFlow fff(c0, c2, reebGraph);
	std::cout << "\n\n\n FRAME FIELD " << (c0-c2).lpNorm<1>() << "\n";
	for (size_t i=0; i<5 && i<static_cast<size_t>(m); ++i){
		std::cout << c0(i,0) << " - " << c2(i,0) << "\n";
	}
	std::vector<cv::Point2d> newlyAddedIntersections;
	std::vector<std::vector<vertex_descriptor>> finalchains, allowedChains;
	std::map<int , int> keypointToValenceMap;
	std::vector<std::pair<int, int>> allowedChainToLabeledPointPair;

	std::vector<MyPolyline> initialPolys; // rough polylines extracted from the graph
	initialPolys = optimizeTopology(reebGraph, pts, bwImg, origMask,covered,fff, roots, newlyAddedIntersections, finalchains, allowedChains, keypointToValenceMap, allowedChainToLabeledPointPair);
	auto distPerVertex = computeGraphDistancesPerPoly(reebGraph, initialPolys, allowedChains);

	for (size_t i=0; i<pts.size(); ++i){
		std::cout << "Labeled point " << i << " has type " << pts_types[i] << "; computed valence is " << keypointToValenceMap[i] << "\n";
	}

	//now run flow on them
	std::vector<MyPolyline> smoothedCurves(initialPolys.size()); // initialPolys after polyvector smoothing

	std::cout << "Starting FLOW in parallel with " << flowIterations << " iterations" << std::endl;
	#pragma omp parallel for
	for (int i = 0; i < initialPolys.size(); ++i)
	{
		MyPolyline pp;
		FrameFieldFlow fff(c0, c2, reebGraph);
		for (int j = 0; j < initialPolys[i].size(); ++j)
			pp.push_back(Eigen::Vector2d(initialPolys[i][j].y(), initialPolys[i][j].x()));

		std::vector<MyPolyline> flowResult = fff.flow(pp, distPerVertex[i], flowIterations, finalchains[i] );
		smoothedCurves[i] = flowResult.back();
	}
	std::cout<< "Done parallel flow \n";

	for (int i = 0; i < smoothedCurves.size(); ++i)
		for (int j = 0; j < smoothedCurves[i].size(); ++j)
			smoothedCurves[i][j] = Eigen::Vector2d(smoothedCurves[i][j].y(), smoothedCurves[i][j].x());

	contractAllButChainEnds(reebGraph, allowedChains);

	std::map<std::pair<int,int>, std::vector<size_t>> pixelToChainCoveringMap = buildPixelToChainsCoveringMap(reebGraph, allowedChains, origMask);
	std::vector<bool> selection;

	std::vector<MyPolyline> selectedPolys; // smooth polylines selected
	selectedPolys = selectBestPolylines(reebGraph, smoothedCurves, initialPolys, finalchains, fff, allowedChainToLabeledPointPair, pts_types, pixelToChainCoveringMap, selection);

	// main() goes on to chop/simplify a copy of these (finalCurves) but writes
	// selectedPolys to OUT.svg; return what it writes.
	return selectedPolys;
}

}  // namespace

using PointsArray = nb::ndarray<const double, nb::shape<-1, 2>, nb::c_contig, nb::device::cpu>;
using LabelsArray = nb::ndarray<const std::int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

NB_MODULE(_flow, module)
{
	module.doc() = "PolyVector Flow (Puhachov et al. 2021), compiled from the vendored "
	               "upstream sources (HiGHS as the solver) and called in-process.";
	module.attr("BACKGROUND_FOREGROUND_THRESHOLD") = BACKGROUND_FOREGROUND_THRESHOLD;

	module.def(
		"vectorize",
		[](const polyvec::GrayImage& image, const PointsArray& points, const LabelsArray& labels,
		   int iterations) {
			if (points.shape(0) != labels.shape(0))
				throw std::invalid_argument("points and labels must have the same length");
			if (iterations < 1)
				throw std::invalid_argument("iterations must be positive");
			FlowInputs in;
			in.gray = polyvec::to_mat(image);
			in.flowIterations = iterations;
			for (size_t i = 0; i < points.shape(0); ++i) {
				// x, y (column, row) -> upstream's (row, column)
				in.pts.emplace_back(points(i, 1), points(i, 0));
				in.pts_types.push_back(static_cast<int>(labels(i)));
			}
			return polyvec::run_captured([&in]() { return vectorize_flow(in); });
		},
		nb::arg("image"), nb::arg("points"), nb::arg("labels"), nb::arg("iterations") = 1500,
		"vectorize(image, points, labels, iterations=1500) -> (strokes, log)\n\n"
		"Run the upstream pipeline on a (H, W) uint8 grayscale image (dark ink on a\n"
		"light background) with keypoints given as an (N, 2) float64 array of x, y\n"
		"pixel coordinates and an (N,) int64 array of labels (1 endpoint, 3 junction,\n"
		"5 sharp corner).  Returns a list of (N, 2) float64 arrays of x, y pixel\n"
		"coordinates (exactly what the command-line program writes to its SVG) and\n"
		"the program's console output.  Raises RuntimeError(message, log) on failure.");
}
