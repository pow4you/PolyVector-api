// gurobi_c++.h — compatibility shim, NOT the real Gurobi header.
//
// The PolyVector-flow sources (vendor/pvflow) are written against the Gurobi
// C++ API, which needs a commercial licence.  This header declares the small
// part of that API the vendored code actually touches, and gurobi_shim.cpp
// implements it with open-source solvers (HiGHS, plus Eigen for the common
// box-bounded least-squares case).  The vendored code compiles against it
// unchanged; build with -DPOLYVEC_SOLVER=gurobi to use real Gurobi instead.
//
// What the vendored code solves through this interface:
//   * SelectCurves.cpp           a binary covering program (which candidate
//                                curves to keep)                 -> HiGHS MIP
//   * GurobiLSLinIneqSolver.cpp  least squares with bounds and optional
//                                linear constraints (stroke widths) -> Eigen / HiGHS QP
//
// Only what is listed here is supported.  Anything else (quadratic
// constraints, callbacks, mixed-integer quadratic objectives, ...) is absent
// on purpose so that new upstream usage fails at compile time, not silently.
#ifndef POLYVEC_GUROBI_SHIM_H_
#define POLYVEC_GUROBI_SHIM_H_

#include <iostream>  // the real header provides it and the vendored code relies on that
#include <memory>
#include <string>
#include <utility>
#include <vector>

#define POLYVEC_GUROBI_SHIM 1

// ---- constants (values match Gurobi's) ------------------------------------
#define GRB_CONTINUOUS 'C'
#define GRB_BINARY 'B'
#define GRB_INTEGER 'I'

#define GRB_LESS_EQUAL '<'
#define GRB_GREATER_EQUAL '>'
#define GRB_EQUAL '='

#define GRB_MINIMIZE 1
#define GRB_MAXIMIZE -1

#define GRB_INFINITY 1e100

// model status codes
#define GRB_LOADED 1
#define GRB_OPTIMAL 2
#define GRB_INFEASIBLE 3
#define GRB_INF_OR_UNBD 4
#define GRB_UNBOUNDED 5
#define GRB_ITERATION_LIMIT 7
#define GRB_TIME_LIMIT 9
#define GRB_NUMERIC 12
#define GRB_SUBOPTIMAL 13

// error codes
#define GRB_ERROR_INVALID_ARGUMENT 10003
#define GRB_ERROR_DATA_NOT_AVAILABLE 10005
#define GRB_ERROR_NOT_SUPPORTED 10017

// Attributes and parameters are distinct enum types so that get()/set()
// overload exactly as they do in Gurobi.
enum GRB_IntAttr { GRB_IntAttr_Status, GRB_IntAttr_NumVars, GRB_IntAttr_NumConstrs };
enum GRB_DoubleAttr { GRB_DoubleAttr_X, GRB_DoubleAttr_ObjVal };
enum GRB_IntParam { GRB_IntParam_OutputFlag, GRB_IntParam_Threads };
enum GRB_DoubleParam {
  GRB_DoubleParam_BarConvTol,
  GRB_DoubleParam_TimeLimit,
  GRB_DoubleParam_MIPGap
};

class GRBModel;

namespace polyvec_gurobi_shim {
struct ModelData;  // defined in gurobi_shim.cpp
}

// ---- exception -------------------------------------------------------------
class GRBException {
 public:
  GRBException(std::string message = "", int error_code = 0)
      : message_(std::move(message)), error_code_(error_code) {}
  const std::string getMessage() const { return message_; }
  int getErrorCode() const { return error_code_; }

 private:
  std::string message_;
  int error_code_;
};

// ---- environment: just a bag of parameters ---------------------------------
class GRBEnv {
 public:
  GRBEnv() = default;
  void set(GRB_IntParam param, int value);
  void set(GRB_DoubleParam param, double value);

 private:
  friend class GRBModel;
  int output_flag_ = 1;
  int threads_ = 0;
  double time_limit_ = GRB_INFINITY;
  double mip_gap_ = 1e-4;
};

// ---- variable: a handle (model, column index) ------------------------------
class GRBVar {
 public:
  GRBVar() = default;
  double get(GRB_DoubleAttr attr) const;
  int index() const { return index_; }

 private:
  friend class GRBModel;
  GRBVar(polyvec_gurobi_shim::ModelData* model, int index) : model_(model), index_(index) {}
  polyvec_gurobi_shim::ModelData* model_ = nullptr;
  int index_ = -1;
};

// ---- linear expression: constant + sum(coeff * var) ------------------------
class GRBLinExpr {
 public:
  GRBLinExpr(double constant = 0.0) : constant_(constant) {}
  GRBLinExpr(GRBVar var, double coeff = 1.0) { terms_.emplace_back(var.index(), coeff); }

  GRBLinExpr& operator+=(const GRBLinExpr& rhs);
  GRBLinExpr& operator-=(const GRBLinExpr& rhs);
  GRBLinExpr& operator*=(double factor);

  // shim-internal helpers (not part of the Gurobi API)
  void addTermByIndex(int column, double coeff) { terms_.emplace_back(column, coeff); }
  const std::vector<std::pair<int, double>>& terms() const { return terms_; }
  double constant() const { return constant_; }

 private:
  std::vector<std::pair<int, double>> terms_;  // (column, coefficient), duplicates allowed
  double constant_ = 0.0;
};

// ---- quadratic expression: linear part + sum(coeff * var * var) ------------
class GRBQuadExpr {
 public:
  struct QuadTerm {
    int row;
    int col;
    double coeff;
  };

  GRBQuadExpr(double constant = 0.0) : linear_(constant) {}
  GRBQuadExpr(const GRBLinExpr& linear) : linear_(linear) {}

  GRBQuadExpr& operator+=(const GRBQuadExpr& rhs);
  GRBQuadExpr& operator+=(const GRBLinExpr& rhs);
  void addQuadTerm(int row, int col, double coeff) { quad_terms_.push_back({row, col, coeff}); }

  // shim-internal accessors
  const std::vector<QuadTerm>& quadTerms() const { return quad_terms_; }
  const GRBLinExpr& linear() const { return linear_; }

 private:
  std::vector<QuadTerm> quad_terms_;
  GRBLinExpr linear_;
};

// Arithmetic used by the vendored code.  Overloads on GRBVar are spelled out
// so that e.g. `0.5 * x * y` resolves without ambiguous implicit conversions.
GRBLinExpr operator*(double factor, GRBVar var);
GRBLinExpr operator*(GRBVar var, double factor);
GRBLinExpr operator*(double factor, const GRBLinExpr& expr);
GRBLinExpr operator*(const GRBLinExpr& expr, double factor);
GRBLinExpr operator+(const GRBLinExpr& lhs, const GRBLinExpr& rhs);
GRBLinExpr operator-(const GRBLinExpr& lhs, const GRBLinExpr& rhs);
GRBQuadExpr operator*(GRBVar lhs, GRBVar rhs);
GRBQuadExpr operator*(const GRBLinExpr& lhs, GRBVar rhs);
GRBQuadExpr operator*(GRBVar lhs, const GRBLinExpr& rhs);
GRBQuadExpr operator*(const GRBLinExpr& lhs, const GRBLinExpr& rhs);
GRBQuadExpr operator+(const GRBQuadExpr& lhs, const GRBQuadExpr& rhs);

// ---- model -----------------------------------------------------------------
class GRBModel {
 public:
  explicit GRBModel(const GRBEnv& env);
  ~GRBModel();
  GRBModel(const GRBModel&) = delete;
  GRBModel& operator=(const GRBModel&) = delete;

  // Both overloads return a new[] array the caller owns, as in Gurobi.
  GRBVar* addVars(int count, char type = GRB_CONTINUOUS);
  GRBVar* addVars(const double* lb, const double* ub, const double* obj, const char* type,
                  const std::string* names, int count);

  void addConstr(const GRBLinExpr& lhs, char sense, double rhs, const std::string& name = "");

  void setObjective(const GRBLinExpr& objective, int sense = GRB_MINIMIZE);
  void setObjective(const GRBQuadExpr& objective, int sense = GRB_MINIMIZE);

  void update() {}                     // Gurobi's lazy-update flush; nothing to do here
  void write(const std::string&) {}    // upstream dumps debug .lp files; not reproduced
  void optimize();

  int get(GRB_IntAttr attr) const;
  double get(GRB_DoubleAttr attr) const;

 private:
  std::unique_ptr<polyvec_gurobi_shim::ModelData> data_;
};

#endif  // POLYVEC_GUROBI_SHIM_H_
