#pragma once

#include <Eigen/Dense>

// 4 -> 8 (ReLU) -> 3 (softmax). Row-vector convention, same as the Python side:
// Z1 = X * W1 + b1, H = relu(Z1), Z2 = H * W2 + b2, P = softmax(Z2).
struct Params {
    Eigen::MatrixXd W1;  // (n_in, n_hidden)
    Eigen::RowVectorXd b1;
    Eigen::MatrixXd W2;  // (n_hidden, n_out)
    Eigen::RowVectorXd b2;
};

struct Forward {
    Eigen::MatrixXd Z1, H, logits, probs;
    Eigen::MatrixXd log_probs;
};

inline Forward forward(const Params& p, const Eigen::MatrixXd& X) {
    Forward f;
    f.Z1 = (X * p.W1).rowwise() + p.b1;
    f.H = f.Z1.cwiseMax(0.0);
    f.logits = (f.H * p.W2).rowwise() + p.b2;
    // Stable log-softmax: subtract the row max before exponentiating.
    Eigen::VectorXd row_max = f.logits.rowwise().maxCoeff();
    Eigen::MatrixXd shifted = f.logits.colwise() - row_max;
    Eigen::VectorXd log_sum = shifted.array().exp().rowwise().sum().log();
    f.log_probs = shifted.colwise() - log_sum;
    f.probs = f.log_probs.array().exp();
    return f;
}

// Mean cross-entropy; labels are class indices.
inline double cross_entropy(const Eigen::MatrixXd& log_probs, const Eigen::VectorXi& labels) {
    double sum = 0.0;
    for (Eigen::Index i = 0; i < log_probs.rows(); ++i) sum -= log_probs(i, labels(i));
    return sum / static_cast<double>(log_probs.rows());
}

inline Eigen::VectorXi predict(const Eigen::MatrixXd& probs) {
    Eigen::VectorXi pred(probs.rows());
    for (Eigen::Index i = 0; i < probs.rows(); ++i) probs.row(i).maxCoeff(&pred(i));
    return pred;
}
