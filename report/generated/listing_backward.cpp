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
