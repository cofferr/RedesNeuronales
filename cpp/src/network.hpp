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

// Gradients have exactly the shape of the parameters, so they are allocated once.
using Grads = Params;

inline Grads make_grads(const Params& p) {
    Grads g;
    g.W1.setZero(p.W1.rows(), p.W1.cols());
    g.b1.setZero(p.b1.size());
    g.W2.setZero(p.W2.rows(), p.W2.cols());
    g.b2.setZero(p.b2.size());
    return g;
}

// Every intermediate of forward + backward lives here and is sized once, up front.
// Why: a training loop runs thousands of epochs; if each epoch created new matrices,
// Eigen would call malloc/free for every temporary. With preallocated buffers each
// assignment below reuses existing memory (same size => no reallocation), and
// `.noalias()` tells Eigen the destination does not overlap the product operands, so
// it writes the product straight into the buffer instead of via a hidden temporary.
struct Workspace {
    Eigen::MatrixXd Z1, H, logits, log_probs, probs;  // forward (n x ...)
    Eigen::MatrixXd dZ2, dH, dZ1;                     // backward (n x ...)
    Eigen::VectorXd row_max, log_sum;

    Workspace(Eigen::Index n, const Params& p)
        : Z1(n, p.W1.cols()),
          H(n, p.W1.cols()),
          logits(n, p.W2.cols()),
          log_probs(n, p.W2.cols()),
          probs(n, p.W2.cols()),
          dZ2(n, p.W2.cols()),
          dH(n, p.W1.cols()),
          dZ1(n, p.W1.cols()),
          row_max(n),
          log_sum(n) {}
};

// Forward pass; results are left in ws (ws.log_probs, ws.probs, ...).
inline void forward(const Params& p, const Eigen::MatrixXd& X, Workspace& ws) {
    ws.Z1.noalias() = X * p.W1;
    ws.Z1.rowwise() += p.b1;
    ws.H = ws.Z1.cwiseMax(0.0);  // ReLU
    ws.logits.noalias() = ws.H * p.W2;
    ws.logits.rowwise() += p.b2;
    // Stable log-softmax: subtracting the row max avoids exp() overflow.
    ws.row_max = ws.logits.rowwise().maxCoeff();
    ws.log_probs = ws.logits.colwise() - ws.row_max;
    ws.log_sum = ws.log_probs.array().exp().rowwise().sum().log();
    ws.log_probs.colwise() -= ws.log_sum;
    ws.probs = ws.log_probs.array().exp();
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

// Backpropagation. Needs forward(p, X, ws) to have been called with the same p and X.
// Loss L = mean_i CE(softmax(z2_i), y_i), so:
//   dZ2 = (P - Y) / N          softmax + cross-entropy collapse to this (Y is one-hot)
//   dW2 = H^T dZ2,  db2 = column sums of dZ2
//   dH  = dZ2 W2^T
//   dZ1 = dH .* (Z1 > 0)       ReLU derivative: 1 where z > 0, else 0
//   dW1 = X^T dZ1,  db1 = column sums of dZ1
inline void backward(const Params& p, const Eigen::MatrixXd& X, const Eigen::VectorXi& labels,
                     Workspace& ws, Grads& g) {
    const double n = static_cast<double>(X.rows());
    ws.dZ2 = ws.probs;
    for (Eigen::Index i = 0; i < ws.dZ2.rows(); ++i) ws.dZ2(i, labels(i)) -= 1.0;  // P - Y
    ws.dZ2 /= n;

    g.W2.noalias() = ws.H.transpose() * ws.dZ2;
    g.b2 = ws.dZ2.colwise().sum();

    ws.dH.noalias() = ws.dZ2 * p.W2.transpose();
    ws.dZ1 = (ws.Z1.array() > 0.0).select(ws.dH, 0.0);

    g.W1.noalias() = X.transpose() * ws.dZ1;
    g.b1 = ws.dZ1.colwise().sum();
}

// Plain gradient descent update: theta <- theta - lr * dtheta (in place).
inline void sgd_step(Params& p, const Grads& g, double lr) {
    p.W1 -= lr * g.W1;
    p.b1 -= lr * g.b1;
    p.W2 -= lr * g.W2;
    p.b2 -= lr * g.b2;
}

// One full-batch epoch. Returns the loss BEFORE the update, which is what sklearn's
// loss_curve_ records, so the two curves are directly comparable.
inline double train_epoch(Params& p, const Eigen::MatrixXd& X, const Eigen::VectorXi& labels,
                          double lr, Workspace& ws, Grads& g) {
    forward(p, X, ws);
    const double loss = cross_entropy(ws.log_probs, labels);
    backward(p, X, labels, ws, g);
    sgd_step(p, g, lr);
    return loss;
}
