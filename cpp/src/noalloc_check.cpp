// Checks that the training loop never touches the heap once the buffers exist.
// EIGEN_RUNTIME_NO_MALLOC makes Eigen abort (assert) on any allocation while
// set_is_malloc_allowed(false) is in effect. Built with asserts enabled (-UNDEBUG),
// otherwise the check would be compiled out.
#define EIGEN_RUNTIME_NO_MALLOC

#include <cstdlib>
#include <iostream>
#include <string>

#include "csv.hpp"
#include "network.hpp"

#ifndef PROJECT_DATA_DIR
#define PROJECT_DATA_DIR "data"
#endif

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : PROJECT_DATA_DIR;
    try {
        const auto cfg = csv::read_named_values(dir + "/config.csv");
        const int epochs = static_cast<int>(cfg.at("epochs"));
        const Eigen::MatrixXd m = csv::read_matrix(dir + "/iris_train.csv", true);
        const Eigen::MatrixXd X = m.leftCols(m.cols() - 1);
        const Eigen::VectorXi y = m.col(m.cols() - 1).cast<int>();
        Params p;
        p.W1 = csv::read_matrix(dir + "/init_w1.csv", false);
        p.b1 = csv::read_matrix(dir + "/init_b1.csv", false).row(0);
        p.W2 = csv::read_matrix(dir + "/init_w2.csv", false);
        p.b2 = csv::read_matrix(dir + "/init_b2.csv", false).row(0);
        Workspace ws(X.rows(), p);
        Grads g = make_grads(p);

        Eigen::internal::set_is_malloc_allowed(false);  // any Eigen allocation aborts now
        for (int e = 0; e < epochs; ++e) train_epoch(p, X, y, cfg.at("learning_rate"), ws, g);
        Eigen::internal::set_is_malloc_allowed(true);

        std::cout << epochs << " epochs without heap allocations in Eigen: PASS\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
