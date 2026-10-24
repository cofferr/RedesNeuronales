#include <Eigen/Dense>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

#include "csv.hpp"
#include "network.hpp"

#ifndef PROJECT_DATA_DIR
#define PROJECT_DATA_DIR "data"
#endif

namespace {

struct Dataset {
    Eigen::MatrixXd X;
    Eigen::VectorXi y;
};

Dataset load_dataset(const std::string& path) {
    Eigen::MatrixXd m = csv::read_matrix(path, true);
    Dataset d;
    d.X = m.leftCols(m.cols() - 1);
    d.y = m.col(m.cols() - 1).cast<int>();
    return d;
}

bool check(const std::string& name, double got, double expected, double tol) {
    double diff = std::abs(got - expected);
    bool ok = diff <= tol;
    std::cout << (ok ? "[ok]   " : "[FAIL] ") << name << ": " << got << " (expected "
              << expected << ", |diff| = " << diff << ")\n";
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : PROJECT_DATA_DIR;
    try {
        std::cout << "Eigen " << EIGEN_WORLD_VERSION << "." << EIGEN_MAJOR_VERSION << "."
                  << EIGEN_MINOR_VERSION << ", data dir: " << dir << "\n";

        const auto cfg = csv::read_named_values(dir + "/config.csv");
        std::cout << "learning_rate = " << cfg.at("learning_rate")
                  << ", epochs = " << static_cast<int>(cfg.at("epochs")) << "\n";

        const Dataset train = load_dataset(dir + "/iris_train.csv");
        const Dataset test = load_dataset(dir + "/iris_test.csv");

        Params p;
        p.W1 = csv::read_matrix(dir + "/init_w1.csv", false);
        p.b1 = csv::read_matrix(dir + "/init_b1.csv", false).row(0);
        p.W2 = csv::read_matrix(dir + "/init_w2.csv", false);
        p.b2 = csv::read_matrix(dir + "/init_b2.csv", false).row(0);

        if (train.X.cols() != cfg.at("n_features") || p.W1.rows() != cfg.at("n_features") ||
            p.W1.cols() != cfg.at("n_hidden") || p.W2.cols() != cfg.at("n_classes") ||
            train.X.rows() != cfg.at("n_train"))
            throw std::runtime_error("data/weights shapes do not match config.csv");

        const Forward ftrain = forward(p, train.X);
        const Forward ftest = forward(p, test.X);
        const auto ref = csv::read_named_values(dir + "/init_reference.csv");

        const double tol = 1e-12;
        bool ok = true;
        ok &= check("initial train loss", cross_entropy(ftrain.log_probs, train.y),
                    ref.at("initial_train_loss"), tol);
        ok &= check("initial test loss", cross_entropy(ftest.log_probs, test.y),
                    ref.at("initial_test_loss"), tol);

        // Probabilities and predictions against sklearn's forward with the same weights.
        for (const auto& [name, f] : {std::pair{"train", &ftrain}, std::pair{"test", &ftest}}) {
            const Eigen::MatrixXd sk = csv::read_matrix(dir + "/init_proba_" + name + ".csv", true);
            ok &= check(std::string("max |proba - sklearn| (") + name + ")",
                        (f->probs - sk).cwiseAbs().maxCoeff(), 0.0, tol);
            const bool same = predict(f->probs) == predict(sk);
            std::cout << (same ? "[ok]   " : "[FAIL] ") << "predictions match sklearn (" << name
                      << ")\n";
            ok &= same;
        }

        std::cout << (ok ? "forward check: PASS\n" : "forward check: FAIL\n");
        return ok ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
