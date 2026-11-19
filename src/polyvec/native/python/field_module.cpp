// polyvec._native._field: PolyVector Fields (Bessmeltsev & Solomon 2019) in-process.
//
// The vendored program (../../vendor/pv/src) is compiled unchanged.  Its
// main.cpp is included below with `main` renamed, which brings in upstream's
// globals and helper routines (calculateGradient, calculateWeight,
// repairMask, ...).  vectorize_field() then runs the same pipeline as that
// main(), step for step, with two differences only:
//
//   * the image comes from a NumPy array instead of imread(FILE), and
//   * the final polylines are returned instead of being written to FILE.svg
//     (and the Qt debug viewer, which main() only feeds, is left out).
//
// Every global main() relies on (bwImg, origMask, m, n, g, tau, weight, ...)
// is reassigned at the start of each call, so repeated calls are independent.

#include "entry_common.h"

#define main polyvec_upstream_field_main
#include "main.cpp"
#undef main

namespace {

polyvec::Polylines vectorize_field(const cv::Mat& gray)
{
	using namespace cv;

	// main(): imread(FILE, IMREAD_COLOR) + cvtColor(BGR2GRAY) == the gray image.
	bwImg = Scalar(255) - gray;
	m = bwImg.rows;
	n = bwImg.cols; // m: height, n: width

	// fill in the mask
	threshold(bwImg, origMask, BACKGROUND_FOREGROUND_THRESHOLD, 255, THRESH_BINARY);

	for (int i = 0; i < 3; ++i)
		repairMask(origMask);

	calculateGradient();
	calculateWeight();

	auto componentMasks = computeComponentMasks(origMask);
	std::cout << "Found " << componentMasks.size() << " connected component(s)." << std::endl;

	std::vector<MyPolyline> polys;
	Eigen::MatrixXcd zeroMatrix(m, n); zeroMatrix.setZero();
	std::array<Eigen::MatrixXcd, 2> roots = { zeroMatrix, zeroMatrix };
	G origGraph, singContractedGraph, singVertsGraph, contractedGraph, cutGraph, optimizedGraph;
	std::vector<MyPolyline> vectorization, newVectorization;

	for (size_t compIdx = 0; compIdx < componentMasks.size(); ++compIdx)
	{
		std::cout << "COMPONENT " << compIdx << " / " << componentMasks.size() << std::endl;
		cv::Mat& compMask = componentMasks[compIdx];
		int nnz = 0;
		auto indices = calculateIndices(compMask, nnz);

		std::cout << "Optimizing...";
		Eigen::VectorXcd X = optimizeByLinearSolve(bwImg, weight, tau, FRAME_FIELD_SMOOTHNESS_WEIGHT, compMask, indices);
		if (X.size() == 0) //if the linear solver failed, use iterative
		{
			X = optimize(bwImg, weight, tau, FRAME_FIELD_SMOOTHNESS_WEIGHT, compMask, indices);
		}

		std::cout << "done. " << std::endl;

		std::cout << "Finding roots.. ";
		auto compRoots = findRoots(X, compMask);

		auto singularities = findSingularities(compRoots, X, indices, compMask);
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

			X = optimizeByLinearSolve(bwImg, weight, tau, FRAME_FIELD_SMOOTHNESS_WEIGHT, compMask, indices);
			if (X.size() == 0) //if the linear solver failed, use iterative
			{
				X = optimize(bwImg, weight, tau, FRAME_FIELD_SMOOTHNESS_WEIGHT, compMask, indices);
			}
			compRoots = findRoots(X, compMask);
			singularities = findSingularities(compRoots, X, indices, compMask);

			std::cout << "done (" << origSingularityCount - (int)singularities.size() << " singularities removed)" << std::endl;
			improved = origSingularityCount - singularities.size() > 0;
		} while (improved);

		roots[0] += compRoots[0]; roots[1] += compRoots[1];

		std::cout << "Tracing... ";
		std::map<std::array<int, 2>, std::vector<PixelInfo>> pixelInfo;
		std::vector<std::array<bool, 2>> endedWithASingularity;
		auto compPolys = traceAll(bwImg, compMask, compMask, compRoots, X, indices, pixelInfo, endedWithASingularity);
		polys.insert(polys.end(), compPolys.begin(), compPolys.end());

		auto reebGraph = computeAlmostReebGraph(compMask, compRoots, compPolys, pixelInfo, singularities, indices, X, endedWithASingularity);
		origGraph = concatenateGraphs(origGraph, reebGraph);

		contractSingularityBranches(reebGraph);
		simpleThresholds(reebGraph);

		connectStuffAroundSingularities(reebGraph, compMask, compPolys, singularities, compRoots, endedWithASingularity);
		singVertsGraph = concatenateGraphs(singVertsGraph, reebGraph);

		edge_iter eit, eend;
		for (std::tie(eit, eend) = boost::edges(reebGraph); eit != eend; ++eit)
		{
			reebGraph[*eit].weight = 1;
		}
		contractLoops(reebGraph, compMask, compPolys);
		contractedGraph = concatenateGraphs(contractedGraph, reebGraph);

		std::map<edge_descriptor, size_t> ignore;
		removeBranchesFilter1(reebGraph, false, ignore);
		cutGraph = concatenateGraphs(cutGraph, reebGraph);

		splitEmUpCorrectly(reebGraph);

		bool chopped = true;
		while (chopped)
		{
			chopped = false;
			for (size_t v = 0; v < boost::num_vertices(reebGraph); ++v)
			{
				if ((boost::degree(v, reebGraph) == 2) && (reebGraph[v].nextToSingularity || reebGraph[v].clusterCurveHitSingularity))
				{
					std::vector<size_t> verts;
					oedge_iter eit, eend;
					for (std::tie(eit, eend) = boost::out_edges(v, reebGraph); eit != eend; ++eit)
					{
						verts.push_back(eit->m_target);
					}
					boost::clear_vertex(v, reebGraph);
					auto e = boost::add_edge(verts[0], verts[1], reebGraph);
					reebGraph[e.first].edgeCurve = -1;
				}
				if (chopped)
					break;
			}
		}

		chopped = true;
		while (chopped)
		{
			chopped = false;
			edge_iter eit, eend;
			for (std::tie(eit, eend) = boost::edges(reebGraph); eit != eend; ++eit)
			{
				if (boost::degree(eit->m_source, reebGraph) > 2 && boost::degree(eit->m_target, reebGraph) > 2)
				{
					boost::remove_edge(*eit, reebGraph);
					chopped = true;
					break;
				}
			}
		}

		std::vector<MyPolyline> compVectorization;
		std::vector<std::vector<double>> radii;
		std::vector<std::array<bool, 2>> protectedEnds;
		std::vector<std::pair<PointOnCurve, PointOnCurve>> yJunctions; //stores info about Y-junctions (where which curve connects)
		std::vector<std::array<bool, 2>> isItASpecialDeg2Vertex;
		std::tie(compVectorization, radii, protectedEnds, isItASpecialDeg2Vertex, yJunctions) = topoGraphEmbedding(reebGraph, compPolys, bwImg);

		optimizedGraph = concatenateGraphs(optimizedGraph, reebGraph);
		G wG;
		std::tie(compVectorization, wG) = chopFakeEnds(compVectorization, radii, protectedEnds, isItASpecialDeg2Vertex, yJunctions);
		vectorization.insert(vectorization.end(), compVectorization.begin(), compVectorization.end());

		for (std::tie(eit, eend) = boost::edges(wG); eit != eend; ++eit)
		{
			wG[*eit].weight = 1.0;
		}

		std::cout << "Finding cycles: ";
		std::vector<edge_descriptor> removedEdges;
		if (boost::num_edges(wG) < 350)
		{
			std::cout << "Using Tarjan's algorithm " << std::endl;
			removedEdges = contractLoops2(wG, compMask, compVectorization);
		}
		else
		{
			std::cout << "Using min spanning trees algorithm " << std::endl;
			removedEdges = contractLoops(wG, compMask, compVectorization);
		}
		std::vector<std::vector<std::pair<double, double>>> cutThosePieces(compVectorization.size());
		for (auto e : removedEdges)
		{
			int curve = wG[e.m_source].clusterPoints[0].curve;
			if (curve == wG[e.m_target].clusterPoints[0].curve)
			{
				double s1 = wG[e.m_source].clusterPoints[0].segmentIdx, s2 = wG[e.m_target].clusterPoints[0].segmentIdx;
				cutThosePieces[curve].push_back(std::minmax(s1, s2));
			}
		}

		decltype(compVectorization) compNewVectorization;

		for (int i = 0; i < compVectorization.size(); ++i)
		{
			if (compVectorization[i].empty())
				continue;

			std::vector<std::pair<double, double>> segments;
			std::sort(cutThosePieces[i].begin(), cutThosePieces[i].end(), [](const std::pair<double, double>& p1, const std::pair<double, double>& p2) {return p1.first < p2.first; });
			segments.push_back({ 0.0,0.0 });
			for (int j = 0; j < cutThosePieces[i].size(); ++j)
			{
				segments.back().second = cutThosePieces[i][j].first;
				segments.push_back({ cutThosePieces[i][j].second,0.0 });
			}
			segments.back().second = compVectorization[i].size() - 1;

			for (int j = 0; j < segments.size(); ++j)
			{
				MyPolyline newPoly;
				if (fabs(segments[j].second - segments[j].first) > 1e-5)
				{
					for (int k = segments[j].first; k <= segments[j].second; ++k)
					{
						if (newPoly.empty() || (newPoly.back() - compVectorization[i][k]).norm() > 1e-6)
							newPoly.push_back(compVectorization[i][k]);
					}
				}
				if (!newPoly.empty())
					compNewVectorization.push_back(newPoly);
			}
		}

		for (int i = 0; i < compNewVectorization.size(); ++i)
			compNewVectorization[i] = simplify(compNewVectorization[i], 1e-2);

		smooth(compNewVectorization);

		newVectorization.insert(newVectorization.end(), compNewVectorization.begin(), compNewVectorization.end());
	}

	// main() now writes newVectorization to FILE.svg; return it instead.
	return newVectorization;
}

}  // namespace

NB_MODULE(_field, module)
{
	module.doc() = "PolyVector Fields (Bessmeltsev & Solomon 2019), compiled from the "
	               "vendored upstream sources and called in-process.";
	module.attr("BACKGROUND_FOREGROUND_THRESHOLD") = BACKGROUND_FOREGROUND_THRESHOLD;

	module.def(
		"vectorize",
		[](const polyvec::GrayImage& image) {
			cv::Mat gray = polyvec::to_mat(image);
			return polyvec::run_captured([&gray]() { return vectorize_field(gray); });
		},
		nb::arg("image"),
		"vectorize(image) -> (strokes, log)\n\n"
		"Run the upstream pipeline on a (H, W) uint8 grayscale image (dark ink on a\n"
		"light background).  Returns a list of (N, 2) float64 arrays of x, y pixel\n"
		"coordinates (exactly what the command-line program writes to its SVG) and\n"
		"the program's console output.  Raises RuntimeError(message, log) on failure.");
}
