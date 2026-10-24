#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <iostream>

#include "network.hpp"

namespace gradcheck {

// Central difference of the loss w.r.t. every entry of `param` (a view into p):
//   dL/dtheta ~ (L(theta + eps) - L(theta - eps)) / (2 eps), error O(eps^2).
template <typename Mat>
Mat numeric_gradient(Params& p, Mat& param, const Eigen::MatrixXd& X,
                     const Eigen::VectorXi& y, Workspace& ws, double eps) {
    Mat g(param.rows(), param.cols());
    for (Eigen::Index i = 0; i < param.rows(); ++i) {
        for (Eigen::Index j = 0; j < param.cols(); ++j) {
            const double saved = param(i, j);
            param(i, j) = saved + eps;
            forward(p, X, ws);
            const double plus = cross_entropy(ws.log_probs, y);
            param(i, j) = saved - eps;
            forward(p, X, ws);
            const double minus = cross_entropy(ws.log_probs, y);
            param(i, j) = saved;
            g(i, j) = (plus - minus) / (2.0 * eps);
        }
    }
    return g;
}

// Relative error ||a - n|| / (||a|| + ||n||) per parameter tensor. Entry-wise relative
// error is unstable for gradients near zero (the denominator vanishes), the norm form is not.
template <typename Mat>
bool check_one(const char* name, Params& p, Mat& param, const Mat& analytic,
               const Eigen::MatrixXd& X, const Eigen::VectorXi& y, Workspace& ws, double eps,
               double tol) {
    const Mat numeric = numeric_gradient(p, param, X, y, ws, eps);
    const double denom = analytic.norm() + numeric.norm();
    const double rel = denom == 0.0 ? 0.0 : (analytic - numeric).norm() / denom;
    const bool ok = rel < tol;
    std::cout << (ok ? "[ok]   " : "[FAIL] ") << name << ": relative error " << rel
              << ", max |diff| " << (analytic - numeric).cwiseAbs().maxCoeff() << "\n";
    return ok;
}

// Compares the analytic gradients (backward) with numeric ones at the current params.
inline bool run(Params p, const Eigen::MatrixXd& X, const Eigen::VectorXi& y, double eps = 1e-6,
                double tol = 1e-6) {
    Workspace ws(X.rows(), p);
    Grads g = make_grads(p);
    forward(p, X, ws);
    backward(p, X, y, ws, g);
    bool ok = true;
    ok &= check_one("dW1", p, p.W1, g.W1, X, y, ws, eps, tol);
    ok &= check_one("db1", p, p.b1, g.b1, X, y, ws, eps, tol);
    ok &= check_one("dW2", p, p.W2, g.W2, X, y, ws, eps, tol);
    ok &= check_one("db2", p, p.b2, g.b2, X, y, ws, eps, tol);
    return ok;
}

}  // namespace gradcheck
