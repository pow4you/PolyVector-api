// Implementation of the Gurobi-compatible shim declared in gurobi_c++.h.
//
// A GRBModel here is a plain in-memory description of the problem.  optimize()
// classifies it and hands it to the cheapest solver that is exact for it:
//
//   linear / mixed-integer linear ............ HiGHS
//   convex quadratic, bounds only ............ sparse Cholesky (Eigen); accepted
//                                              only if the unconstrained optimum
//                                              already satisfies every bound
//   convex quadratic, anything else .......... HiGHS QP
//
// The Cholesky shortcut matters: the stroke-width solve in the vendored code is
// a least-squares problem over every graph vertex whose bounds are never
// active, and an active-set QP solver is needlessly slow on it.
#include "gurobi_c++.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <tuple>

#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include "Highs.h"

#ifdef POLYVEC_SHIM_LOG_TO_COUT
#include <iostream>
#endif

namespace polyvec_gurobi_shim {

using Terms = std::vector<std::pair<int, double>>;

struct Constraint {
  Terms terms;   // merged, sorted by column, zeros dropped
  char sense;    // GRB_LESS_EQUAL / GRB_GREATER_EQUAL / GRB_EQUAL
  double rhs;    // lhs constant already moved to this side

  bool operator<(const Constraint& other) const {
    return std::tie(sense, rhs, terms) < std::tie(other.sense, other.rhs, other.terms);
  }
};

struct ModelData {
  // parameters, copied from the GRBEnv when the model is created (as Gurobi does)
  int output_flag = 1;
  int threads = 0;
  double time_limit = GRB_INFINITY;
  double mip_gap = 1e-4;

  // variables
  std::vector<double> lb, ub;
  std::vector<char> type;

  // constraints and objective
  std::vector<Constraint> rows;
  std::vector<GRBQuadExpr::QuadTerm> quad;
  Terms linear;
  double offset = 0.0;
  int sense = GRB_MINIMIZE;

  // result
  int status = GRB_LOADED;
  bool has_solution = false;
  std::vector<double> x;
  double objval = 0.0;

  int numVars() const { return static_cast<int>(lb.size()); }
};

namespace {

// Sum duplicate columns, drop zeros, sort by column.
Terms mergeTerms(const Terms& terms) {
  Terms merged = terms;
  std::sort(merged.begin(), merged.end(),
            [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
              return a.first < b.first;
            });
  Terms out;
  for (const auto& term : merged) {
    if (!out.empty() && out.back().first == term.first)
      out.back().second += term.second;
    else
      out.push_back(term);
  }
  out.erase(std::remove_if(out.begin(), out.end(),
                           [](const std::pair<int, double>& t) { return t.second == 0.0; }),
            out.end());
  return out;
}

double toHighsBound(double value) {
  if (value >= 1e20) return kHighsInf;
  if (value <= -1e20) return -kHighsInf;
  return value;
}

// Hessian Q of the objective  c'x + 1/2 x'Qx  given  sum(coeff * x_row * x_col).
Eigen::SparseMatrix<double> buildHessian(const ModelData& m) {
  std::vector<Eigen::Triplet<double>> triplets;
  triplets.reserve(2 * m.quad.size());
  for (const auto& q : m.quad) {
    if (q.row == q.col) {
      triplets.emplace_back(q.row, q.col, 2.0 * q.coeff);
    } else {
      triplets.emplace_back(q.row, q.col, q.coeff);
      triplets.emplace_back(q.col, q.row, q.coeff);
    }
  }
  Eigen::SparseMatrix<double> hessian(m.numVars(), m.numVars());
  hessian.setFromTriplets(triplets.begin(), triplets.end());  // sums duplicates
  hessian.prune(0.0);
  return hessian;
}

std::vector<double> denseCost(const ModelData& m) {
  std::vector<double> cost(m.numVars(), 0.0);
  for (const auto& term : m.linear) cost[term.first] += term.second;
  return cost;
}

double evaluateObjective(const ModelData& m, const std::vector<double>& x) {
  double value = m.offset;
  for (const auto& term : m.linear) value += term.second * x[term.first];
  for (const auto& q : m.quad) value += q.coeff * x[q.row] * x[q.col];
  return value;
}

void storeSolution(ModelData& m, std::vector<double> x, int status) {
  for (int i = 0; i < m.numVars(); ++i)
    if (m.type[i] != GRB_CONTINUOUS) x[i] = std::round(x[i]);  // upstream tests `== 1`
  m.x = std::move(x);
  m.objval = evaluateObjective(m, m.x);
  m.has_solution = true;
  m.status = status;
}

// Bounds-only convex QP: solve Qx = -c and accept if x is within the bounds.
bool solveBoxQpDirect(ModelData& m, const Eigen::SparseMatrix<double>& hessian,
                      const std::vector<double>& cost) {
  const int n = m.numVars();
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver(hessian);
  if (solver.info() != Eigen::Success) return false;
  if (solver.vectorD().minCoeff() <= 0.0) return false;  // not positive definite

  Eigen::VectorXd rhs(n);
  for (int i = 0; i < n; ++i) rhs[i] = -cost[i];
  const Eigen::VectorXd solution = solver.solve(rhs);
  if (solver.info() != Eigen::Success || !solution.allFinite()) return false;

  std::vector<double> x(n);
  for (int i = 0; i < n; ++i) {
    const double tol = 1e-9 * std::max(1.0, std::abs(solution[i]));
    if (solution[i] < m.lb[i] - tol || solution[i] > m.ub[i] + tol) return false;
    x[i] = std::min(std::max(solution[i], m.lb[i]), m.ub[i]);
  }
  storeSolution(m, std::move(x), GRB_OPTIMAL);
  return true;
}

int toGurobiStatus(HighsModelStatus status) {
  switch (status) {
    case HighsModelStatus::kOptimal:
      return GRB_OPTIMAL;
    case HighsModelStatus::kInfeasible:
      return GRB_INFEASIBLE;
    case HighsModelStatus::kUnboundedOrInfeasible:
      return GRB_INF_OR_UNBD;
    case HighsModelStatus::kUnbounded:
      return GRB_UNBOUNDED;
    case HighsModelStatus::kTimeLimit:
      return GRB_TIME_LIMIT;
    case HighsModelStatus::kIterationLimit:
      return GRB_ITERATION_LIMIT;
    default:
      return GRB_NUMERIC;
  }
}

void solveWithHighs(ModelData& m, const Eigen::SparseMatrix<double>* hessian,
                    const std::vector<double>& cost, bool has_integers) {
  const int n = m.numVars();

  // Covering programs repeat the same row once per pixel; keep one of each.
  const std::set<Constraint> rows(m.rows.begin(), m.rows.end());

  HighsModel model;
  HighsLp& lp = model.lp_;
  lp.num_col_ = n;
  lp.num_row_ = static_cast<HighsInt>(rows.size());
  lp.sense_ = (m.sense == GRB_MAXIMIZE) ? ObjSense::kMaximize : ObjSense::kMinimize;
  lp.offset_ = m.offset;
  lp.col_cost_ = cost;
  lp.col_lower_.resize(n);
  lp.col_upper_.resize(n);
  for (int i = 0; i < n; ++i) {
    lp.col_lower_[i] = toHighsBound(m.lb[i]);
    lp.col_upper_[i] = toHighsBound(m.ub[i]);
  }

  lp.a_matrix_.format_ = MatrixFormat::kRowwise;
  lp.a_matrix_.num_col_ = lp.num_col_;
  lp.a_matrix_.num_row_ = lp.num_row_;
  lp.a_matrix_.start_.assign(1, 0);
  for (const Constraint& row : rows) {
    for (const auto& term : row.terms) {
      lp.a_matrix_.index_.push_back(term.first);
      lp.a_matrix_.value_.push_back(term.second);
    }
    lp.a_matrix_.start_.push_back(static_cast<HighsInt>(lp.a_matrix_.index_.size()));
    lp.row_lower_.push_back(row.sense == GRB_LESS_EQUAL ? -kHighsInf : row.rhs);
    lp.row_upper_.push_back(row.sense == GRB_GREATER_EQUAL ? kHighsInf : row.rhs);
  }

  if (has_integers) {
    lp.integrality_.resize(n);
    for (int i = 0; i < n; ++i)
      lp.integrality_[i] =
          (m.type[i] == GRB_CONTINUOUS) ? HighsVarType::kContinuous : HighsVarType::kInteger;
  }

  if (hessian != nullptr) {
    HighsHessian& h = model.hessian_;
    h.dim_ = n;
    h.format_ = HessianFormat::kTriangular;  // lower triangle, column-wise
    h.start_.assign(1, 0);
    for (int col = 0; col < hessian->outerSize(); ++col) {
      for (Eigen::SparseMatrix<double>::InnerIterator it(*hessian, col); it; ++it) {
        if (it.row() < col) continue;
        h.index_.push_back(static_cast<HighsInt>(it.row()));
        h.value_.push_back(it.value());
      }
      h.start_.push_back(static_cast<HighsInt>(h.index_.size()));
    }
  }

  Highs highs;
  highs.setOptionValue("output_flag", m.output_flag != 0);
#ifdef POLYVEC_SHIM_LOG_TO_COUT
  // In the Python extension, std::cout is captured per call (the log handed
  // back to Python); HiGHS would otherwise print straight to the console.
  highs.setOptionValue("log_to_console", false);
  // `auto message`: const char* in recent HiGHS (C callback), std::string in
  // older releases (the FetchContent fallback).
  highs.setCallback(
      [](int type, auto message, const HighsCallbackDataOut*,
         HighsCallbackDataIn*, void*) {
        if (type == kCallbackLogging) std::cout << message;
      },
      nullptr);
  highs.startCallback(kCallbackLogging);
#endif
  if (m.threads > 0) highs.setOptionValue("threads", m.threads);
  if (m.time_limit < 1e20) highs.setOptionValue("time_limit", m.time_limit);
  if (has_integers) highs.setOptionValue("mip_rel_gap", m.mip_gap);

  if (highs.passModel(std::move(model)) == HighsStatus::kError ||
      highs.run() == HighsStatus::kError) {
    m.status = GRB_NUMERIC;
    return;
  }

  m.status = toGurobiStatus(highs.getModelStatus());
  const HighsSolution& solution = highs.getSolution();
  if (m.status == GRB_OPTIMAL && solution.value_valid)
    storeSolution(m, solution.col_value, GRB_OPTIMAL);
}

}  // namespace
}  // namespace polyvec_gurobi_shim

using polyvec_gurobi_shim::ModelData;

// ---- GRBEnv -----------------------------------------------------------------
void GRBEnv::set(GRB_IntParam param, int value) {
  switch (param) {
    case GRB_IntParam_OutputFlag:
      output_flag_ = value;
      break;
    case GRB_IntParam_Threads:
      threads_ = value;
      break;
  }
}

void GRBEnv::set(GRB_DoubleParam param, double value) {
  switch (param) {
    case GRB_DoubleParam_TimeLimit:
      time_limit_ = value;
      break;
    case GRB_DoubleParam_MIPGap:
      mip_gap_ = value;
      break;
    case GRB_DoubleParam_BarConvTol:
      break;  // barrier tolerance: no equivalent needed, the solves here are exact
  }
}

// ---- GRBVar -----------------------------------------------------------------
double GRBVar::get(GRB_DoubleAttr attr) const {
  if (attr != GRB_DoubleAttr_X)
    throw GRBException("GRBVar::get: unsupported attribute", GRB_ERROR_NOT_SUPPORTED);
  if (model_ == nullptr || !model_->has_solution)
    throw GRBException("GRBVar::get: no solution available", GRB_ERROR_DATA_NOT_AVAILABLE);
  return model_->x[index_];
}

// ---- expressions ------------------------------------------------------------
GRBLinExpr& GRBLinExpr::operator+=(const GRBLinExpr& rhs) {
  constant_ += rhs.constant_;
  terms_.insert(terms_.end(), rhs.terms_.begin(), rhs.terms_.end());
  return *this;
}

GRBLinExpr& GRBLinExpr::operator-=(const GRBLinExpr& rhs) {
  constant_ -= rhs.constant_;
  terms_.reserve(terms_.size() + rhs.terms_.size());
  for (const auto& term : rhs.terms_) terms_.emplace_back(term.first, -term.second);
  return *this;
}

GRBLinExpr& GRBLinExpr::operator*=(double factor) {
  constant_ *= factor;
  for (auto& term : terms_) term.second *= factor;
  return *this;
}

GRBQuadExpr& GRBQuadExpr::operator+=(const GRBQuadExpr& rhs) {
  quad_terms_.insert(quad_terms_.end(), rhs.quad_terms_.begin(), rhs.quad_terms_.end());
  linear_ += rhs.linear_;
  return *this;
}

GRBQuadExpr& GRBQuadExpr::operator+=(const GRBLinExpr& rhs) {
  linear_ += rhs;
  return *this;
}

GRBLinExpr operator*(double factor, GRBVar var) { return GRBLinExpr(var, factor); }
GRBLinExpr operator*(GRBVar var, double factor) { return GRBLinExpr(var, factor); }

GRBLinExpr operator*(double factor, const GRBLinExpr& expr) {
  GRBLinExpr result = expr;
  result *= factor;
  return result;
}

GRBLinExpr operator*(const GRBLinExpr& expr, double factor) { return factor * expr; }

GRBLinExpr operator+(const GRBLinExpr& lhs, const GRBLinExpr& rhs) {
  GRBLinExpr result = lhs;
  result += rhs;
  return result;
}

GRBLinExpr operator-(const GRBLinExpr& lhs, const GRBLinExpr& rhs) {
  GRBLinExpr result = lhs;
  result -= rhs;
  return result;
}

GRBQuadExpr operator*(const GRBLinExpr& lhs, const GRBLinExpr& rhs) {
  // (a + sum a_i x_i)(b + sum b_j x_j)
  //   = ab + a * sum b_j x_j + b * sum a_i x_i + sum a_i b_j x_i x_j
  GRBLinExpr linear(lhs.constant() * rhs.constant());
  for (const auto& b : rhs.terms()) linear.addTermByIndex(b.first, lhs.constant() * b.second);
  for (const auto& a : lhs.terms()) linear.addTermByIndex(a.first, rhs.constant() * a.second);

  GRBQuadExpr result(linear);
  for (const auto& a : lhs.terms())
    for (const auto& b : rhs.terms()) result.addQuadTerm(a.first, b.first, a.second * b.second);
  return result;
}

GRBQuadExpr operator*(GRBVar lhs, GRBVar rhs) { return GRBLinExpr(lhs) * GRBLinExpr(rhs); }
GRBQuadExpr operator*(const GRBLinExpr& lhs, GRBVar rhs) { return lhs * GRBLinExpr(rhs); }
GRBQuadExpr operator*(GRBVar lhs, const GRBLinExpr& rhs) { return GRBLinExpr(lhs) * rhs; }

GRBQuadExpr operator+(const GRBQuadExpr& lhs, const GRBQuadExpr& rhs) {
  GRBQuadExpr result = lhs;
  result += rhs;
  return result;
}

// ---- GRBModel ---------------------------------------------------------------
GRBModel::GRBModel(const GRBEnv& env) : data_(new ModelData) {
  data_->output_flag = env.output_flag_;
  data_->threads = env.threads_;
  data_->time_limit = env.time_limit_;
  data_->mip_gap = env.mip_gap_;
}

GRBModel::~GRBModel() = default;

GRBVar* GRBModel::addVars(int count, char type) {
  if (count < 0) throw GRBException("addVars: negative count", GRB_ERROR_INVALID_ARGUMENT);
  const std::vector<char> types(static_cast<size_t>(count), type);
  return addVars(nullptr, nullptr, nullptr, types.data(), nullptr, count);
}

GRBVar* GRBModel::addVars(const double* lb, const double* ub, const double* obj, const char* type,
                          const std::string* /*names*/, int count) {
  if (count < 0) throw GRBException("addVars: negative count", GRB_ERROR_INVALID_ARGUMENT);
  GRBVar* vars = new GRBVar[count];
  for (int i = 0; i < count; ++i) {
    const int index = data_->numVars();
    const char var_type = type ? type[i] : GRB_CONTINUOUS;
    // Gurobi defaults: lb = 0, ub = +inf (1 for binaries).
    double lower = lb ? lb[i] : 0.0;
    double upper = ub ? ub[i] : (var_type == GRB_BINARY ? 1.0 : GRB_INFINITY);
    if (var_type == GRB_BINARY) {
      lower = std::max(lower, 0.0);
      upper = std::min(upper, 1.0);
    }
    data_->lb.push_back(lower);
    data_->ub.push_back(upper);
    data_->type.push_back(var_type);
    if (obj && obj[i] != 0.0) data_->linear.emplace_back(index, obj[i]);
    vars[i] = GRBVar(data_.get(), index);
  }
  return vars;
}

void GRBModel::addConstr(const GRBLinExpr& lhs, char sense, double rhs, const std::string&) {
  if (sense != GRB_LESS_EQUAL && sense != GRB_GREATER_EQUAL && sense != GRB_EQUAL)
    throw GRBException("addConstr: unknown sense", GRB_ERROR_INVALID_ARGUMENT);
  data_->rows.push_back({polyvec_gurobi_shim::mergeTerms(lhs.terms()), sense, rhs - lhs.constant()});
}

void GRBModel::setObjective(const GRBLinExpr& objective, int sense) {
  setObjective(GRBQuadExpr(objective), sense);
}

void GRBModel::setObjective(const GRBQuadExpr& objective, int sense) {
  data_->quad = objective.quadTerms();
  data_->linear = objective.linear().terms();
  data_->offset = objective.linear().constant();
  data_->sense = sense;
}

void GRBModel::optimize() {
  using namespace polyvec_gurobi_shim;
  ModelData& m = *data_;
  m.status = GRB_LOADED;
  m.has_solution = false;

  const bool has_integers =
      std::any_of(m.type.begin(), m.type.end(), [](char t) { return t != GRB_CONTINUOUS; });
  const Eigen::SparseMatrix<double> hessian = buildHessian(m);
  const bool has_quadratic = hessian.nonZeros() > 0;
  const std::vector<double> cost = denseCost(m);

  if (has_quadratic && has_integers)
    throw GRBException("mixed-integer quadratic programs are not supported by the HiGHS shim",
                       GRB_ERROR_NOT_SUPPORTED);
  if (has_quadratic && m.sense == GRB_MAXIMIZE)
    throw GRBException("quadratic maximisation is not supported by the HiGHS shim",
                       GRB_ERROR_NOT_SUPPORTED);

  if (has_quadratic && m.rows.empty() && solveBoxQpDirect(m, hessian, cost)) return;
  solveWithHighs(m, has_quadratic ? &hessian : nullptr, cost, has_integers);
}

int GRBModel::get(GRB_IntAttr attr) const {
  switch (attr) {
    case GRB_IntAttr_Status:
      return data_->status;
    case GRB_IntAttr_NumVars:
      return data_->numVars();
    case GRB_IntAttr_NumConstrs:
      return static_cast<int>(data_->rows.size());
  }
  throw GRBException("GRBModel::get: unsupported attribute", GRB_ERROR_NOT_SUPPORTED);
}

double GRBModel::get(GRB_DoubleAttr attr) const {
  if (attr != GRB_DoubleAttr_ObjVal)
    throw GRBException("GRBModel::get: unsupported attribute", GRB_ERROR_NOT_SUPPORTED);
  if (!data_->has_solution)
    throw GRBException("GRBModel::get: no solution available", GRB_ERROR_DATA_NOT_AVAILABLE);
  return data_->objval;
}
