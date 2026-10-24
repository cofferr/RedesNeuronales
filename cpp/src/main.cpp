#include <Eigen/Dense>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <sys/resource.h>
#ifdef __APPLE__
#include <mach/mach.h>
#endif

#include "csv.hpp"
#include "gradcheck.hpp"
#include "network.hpp"

#ifndef PROJECT_DATA_DIR
#define PROJECT_DATA_DIR "data"
#endif
#ifndef PROJECT_RESULTS_DIR
#define PROJECT_RESULTS_DIR "results"
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

Params load_params(const std::string& dir, const std::string& prefix) {
    Params p;
    p.W1 = csv::read_matrix(dir + "/" + prefix + "w1.csv", false);
    p.b1 = csv::read_matrix(dir + "/" + prefix + "b1.csv", false).row(0);
    p.W2 = csv::read_matrix(dir + "/" + prefix + "w2.csv", false);
    p.b2 = csv::read_matrix(dir + "/" + prefix + "b2.csv", false).row(0);
    return p;
}

bool check(const std::string& name, double got, double expected, double tol) {
    double diff = std::abs(got - expected);
    bool ok = diff <= tol;
    std::cout << (ok ? "[ok]   " : "[FAIL] ") << name << ": " << got << " (expected "
              << expected << ", |diff| = " << diff << ")\n";
    return ok;
}

double max_abs_diff(const Params& a, const Params& b) {
    return std::max({(a.W1 - b.W1).cwiseAbs().maxCoeff(), (a.b1 - b.b1).cwiseAbs().maxCoeff(),
                     (a.W2 - b.W2).cwiseAbs().maxCoeff(), (a.b2 - b.b2).cwiseAbs().maxCoeff()});
}

// Forward pass with the initial weights vs the values exported by Python/sklearn.
bool check_forward(const std::string& dir, const Params& p, const Dataset& train,
                   const Dataset& test) {
    Workspace ws_train(train.X.rows(), p), ws_test(test.X.rows(), p);
    forward(p, train.X, ws_train);
    forward(p, test.X, ws_test);
    const auto ref = csv::read_named_values(dir + "/init_reference.csv");
    const double tol = 1e-12;
    bool ok = true;
    ok &= check("initial train loss", cross_entropy(ws_train.log_probs, train.y),
                ref.at("initial_train_loss"), tol);
    ok &= check("initial test loss", cross_entropy(ws_test.log_probs, test.y),
                ref.at("initial_test_loss"), tol);
    for (const auto& [name, ws] : {std::pair{"train", &ws_train}, std::pair{"test", &ws_test}}) {
        const Eigen::MatrixXd sk = csv::read_matrix(dir + "/init_proba_" + name + ".csv", true);
        ok &= check(std::string("max |proba - sklearn| (") + name + ")",
                    (ws->probs - sk).cwiseAbs().maxCoeff(), 0.0, tol);
        const bool same = predict(ws->probs) == predict(sk);
        std::cout << (same ? "[ok]   " : "[FAIL] ") << "predictions match sklearn (" << name
                  << ")\n";
        ok &= same;
    }
    return ok;
}

// Train check_epochs epochs from the initial weights and compare the weights and the
// per-epoch loss with sklearn's (partial_fit, one epoch per call).
bool check_training(const std::string& dir, Params p, const Dataset& train, double lr,
                    int epochs) {
    Workspace ws(train.X.rows(), p);
    Grads g = make_grads(p);
    const Eigen::MatrixXd ref_loss = csv::read_matrix(dir + "/ref_epoch_loss.csv", true);
    double worst_loss_diff = 0.0;
    for (int e = 0; e < epochs; ++e) {
        const double loss = train_epoch(p, train.X, train.y, lr, ws, g);
        worst_loss_diff = std::max(worst_loss_diff, std::abs(loss - ref_loss(e, 1)));
    }
    const Params ref = load_params(dir, "ref_epoch_");
    bool ok = true;
    ok &= check("weights after " + std::to_string(epochs) + " epochs: max |diff|",
                max_abs_diff(p, ref), 0.0, 1e-10);
    ok &= check("loss curve (" + std::to_string(epochs) + " epochs): max |diff|",
                worst_loss_diff, 0.0, 1e-10);
    return ok;
}

double accuracy(const Eigen::MatrixXd& probs, const Eigen::VectorXi& y) {
    return static_cast<double>((predict(probs).array() == y.array()).count()) /
           static_cast<double>(y.size());
}

// Full training run: epochs of full-batch gradient descent from the shared initial weights.
// Writes cpp_loss_curve.csv and cpp_results.json (same field names as the sklearn files).
void run_training(const std::string& out_dir, const std::map<std::string, double>& cfg,
                  Params p, const Dataset& train, const Dataset& test) {
    const double lr = cfg.at("learning_rate");
    const int epochs = static_cast<int>(cfg.at("epochs"));
    Workspace ws(train.X.rows(), p);
    Grads g = make_grads(p);
    std::vector<double> losses(epochs);  // allocated before the clock starts

    // Only the training loop is timed (no file I/O, no final metrics). The loss recorded at
    // epoch e is the one computed in the forward pass BEFORE that epoch's update, the same
    // point where sklearn's loss_curve_ is recorded.
    const auto start = std::chrono::steady_clock::now();
    for (int e = 0; e < epochs; ++e) losses[e] = train_epoch(p, train.X, train.y, lr, ws, g);
    const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;

    // Final metrics with the trained weights.
    Workspace ws_test(test.X.rows(), p);
    forward(p, train.X, ws);
    forward(p, test.X, ws_test);
    const double train_loss = cross_entropy(ws.log_probs, train.y);
    const double test_loss = cross_entropy(ws_test.log_probs, test.y);
    const double train_acc = accuracy(ws.probs, train.y);
    const double test_acc = accuracy(ws_test.probs, test.y);

    std::filesystem::create_directories(out_dir);
    std::ofstream curve(out_dir + "/cpp_loss_curve.csv");
    curve << std::setprecision(17) << "epoch,loss\n";
    for (int e = 0; e < epochs; ++e) curve << e + 1 << "," << losses[e] << "\n";

    std::ofstream json(out_dir + "/cpp_results.json");
    json << std::setprecision(17) << "{\n"
         << "  \"epochs\": " << epochs << ",\n"
         << "  \"learning_rate\": " << lr << ",\n"
         << "  \"train_loss\": " << train_loss << ",\n"
         << "  \"test_loss\": " << test_loss << ",\n"
         << "  \"train_accuracy\": " << train_acc << ",\n"
         << "  \"test_accuracy\": " << test_acc << ",\n"
         << "  \"time_train_s\": " << elapsed.count() << "\n"
         << "}\n";
    if (!curve || !json) throw std::runtime_error("cannot write results to " + out_dir);

    std::cout << "trained " << epochs << " epochs in " << elapsed.count() << " s\n"
              << "train loss " << train_loss << ", acc " << train_acc << "; test loss "
              << test_loss << ", acc " << test_acc << "\nresults written to " << out_dir << "\n";
}

// Peak resident set size of this process so far, in bytes.
// getrusage reports ru_maxrss in BYTES on macOS but in KILOBYTES on Linux.
std::size_t peak_rss_bytes() {
    rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
#ifdef __APPLE__
    return static_cast<std::size_t>(ru.ru_maxrss);
#else
    return static_cast<std::size_t>(ru.ru_maxrss) * 1024;
#endif
}

// Current resident set size in bytes (0 if the platform is not supported).
std::size_t current_rss_bytes() {
#ifdef __APPLE__
    mach_task_basic_info info{};
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, reinterpret_cast<task_info_t>(&info),
                  &count) != KERN_SUCCESS)
        return 0;
    return static_cast<std::size_t>(info.resident_size);
#else
    std::ifstream statm("/proc/self/statm");
    std::size_t pages_total = 0, pages_resident = 0;
    statm >> pages_total >> pages_resident;
    return pages_resident * static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
#endif
}

template <typename T>
void write_array(std::ostream& os, const char* key, const std::vector<T>& v) {
    os << "  \"" << key << "\": [";
    for (std::size_t i = 0; i < v.size(); ++i) os << (i ? ", " : "") << v[i];
    os << "]";
}

// Benchmark: `warmup` discarded + `repeat` measured trainings, all inside this process.
// Each repetition starts from a copy of the initial weights held in memory (the copy is
// made outside the timed region), and only the training loop is timed. After each
// repetition the final train loss is checked against the reference (sklearn).
void run_bench(const std::string& out_dir, const std::map<std::string, double>& cfg,
               const Params& init, const Dataset& train, int warmup, int repeat,
               double ref_loss) {
    const double lr = cfg.at("learning_rate");
    const int epochs = static_cast<int>(cfg.at("epochs"));
    Workspace ws(train.X.rows(), init);
    Grads g = make_grads(init);
    std::vector<double> times, final_losses;
    for (int r = 0; r < warmup + repeat; ++r) {
        Params p = init;  // reload the initial weights from memory (not timed)
        const auto start = std::chrono::steady_clock::now();
        for (int e = 0; e < epochs; ++e) train_epoch(p, train.X, train.y, lr, ws, g);
        const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;

        forward(p, train.X, ws);
        const double loss = cross_entropy(ws.log_probs, train.y);
        if (std::abs(loss - ref_loss) > 1e-9)
            throw std::runtime_error("repetition " + std::to_string(r) + ": final loss " +
                                     std::to_string(loss) + " differs from the reference " +
                                     std::to_string(ref_loss));
        if (r >= warmup) {
            times.push_back(elapsed.count());
            final_losses.push_back(loss);
        }
    }
    std::filesystem::create_directories(out_dir);
    std::ofstream json(out_dir + "/cpp_bench_block.json");
    json << std::setprecision(17) << "{\n"
         << "  \"eigen_version\": \"" << EIGEN_WORLD_VERSION << "." << EIGEN_MAJOR_VERSION
         << "." << EIGEN_MINOR_VERSION << "\",\n"
         << "  \"compiler\": \"" << __VERSION__ << "\",\n"
         << "  \"warmup\": " << warmup << ",\n"
         << "  \"repeat\": " << repeat << ",\n";
    write_array(json, "times_s", times);
    json << ",\n";
    write_array(json, "final_train_losses", final_losses);
    json << "\n}\n";
    if (!json) throw std::runtime_error("cannot write benchmark block to " + out_dir);
}

// Memory run (one clean process, one training). Prints one JSON line on stdout.
// train_increment = growth of the peak RSS during training (high-water mark after minus
// before), so memory acquired while loading the data is not attributed to training.
void run_mem(const std::map<std::string, double>& cfg, const Params& init, const Dataset& train) {
    const double lr = cfg.at("learning_rate");
    const int epochs = static_cast<int>(cfg.at("epochs"));
    Params p = init;
    Workspace ws(train.X.rows(), p);
    Grads g = make_grads(p);
    const std::size_t rss_before = current_rss_bytes(), peak_before = peak_rss_bytes();
    for (int e = 0; e < epochs; ++e) train_epoch(p, train.X, train.y, lr, ws, g);
    const std::size_t peak_after = peak_rss_bytes();
    std::cout << "{\"rss_before_train_bytes\": " << rss_before
              << ", \"peak_before_train_bytes\": " << peak_before
              << ", \"peak_after_train_bytes\": " << peak_after
              << ", \"train_increment_bytes\": " << peak_after - peak_before << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::string dir = PROJECT_DATA_DIR;
    std::string out_dir = PROJECT_RESULTS_DIR;
    bool run_gradcheck = false, run_train = false, run_bench_mode = false, run_mem_mode = false;
    int repeat = 20, warmup = 2;
    double ref_loss = std::nan("");
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--gradcheck") run_gradcheck = true;
        else if (arg == "--train") run_train = true;
        else if (arg == "--bench") run_bench_mode = true;
        else if (arg == "--mem") run_mem_mode = true;
        else if (arg == "--repeat" && i + 1 < argc) repeat = std::stoi(argv[++i]);
        else if (arg == "--warmup" && i + 1 < argc) warmup = std::stoi(argv[++i]);
        else if (arg == "--ref-loss" && i + 1 < argc) ref_loss = std::stod(argv[++i]);
        else if (arg == "--out" && i + 1 < argc) out_dir = argv[++i];
        else dir = arg;
    }
    try {
        // --mem prints a single JSON line, so the usual banner is suppressed in that mode.
        if (!run_mem_mode)
        std::cout << "Eigen " << EIGEN_WORLD_VERSION << "." << EIGEN_MAJOR_VERSION << "."
                  << EIGEN_MINOR_VERSION << ", data dir: " << dir << "\n";

        const auto cfg = csv::read_named_values(dir + "/config.csv");
        const double lr = cfg.at("learning_rate");
        const int check_epochs = static_cast<int>(cfg.at("check_epochs"));
        if (!run_mem_mode)
            std::cout << "learning_rate = " << lr
                      << ", epochs = " << static_cast<int>(cfg.at("epochs")) << "\n";

        const Dataset train = load_dataset(dir + "/iris_train.csv");
        const Dataset test = load_dataset(dir + "/iris_test.csv");
        const Params init = load_params(dir, "init_");

        if (train.X.cols() != cfg.at("n_features") || init.W1.rows() != cfg.at("n_features") ||
            init.W1.cols() != cfg.at("n_hidden") || init.W2.cols() != cfg.at("n_classes") ||
            train.X.rows() != cfg.at("n_train"))
            throw std::runtime_error("data/weights shapes do not match config.csv");

        bool ok = true;
        if (run_mem_mode) {
            run_mem(cfg, init, train);
        } else if (run_bench_mode) {
            if (std::isnan(ref_loss)) throw std::runtime_error("--bench needs --ref-loss <value>");
            run_bench(out_dir, cfg, init, train, warmup, repeat, ref_loss);
            std::cout << "benchmark block done: " << repeat << " repetitions\n";
        } else if (run_train) {
            run_training(out_dir, cfg, init, train, test);
        } else if (run_gradcheck) {
            std::cout << "-- gradient check at the initial weights\n";
            ok &= gradcheck::run(init, train.X, train.y);
            // Second point: after the check epochs the biases are nonzero and ReLU units differ.
            Params trained = load_params(dir, "ref_epoch_");
            std::cout << "-- gradient check after " << check_epochs << " epochs\n";
            ok &= gradcheck::run(trained, train.X, train.y);
            std::cout << (ok ? "gradient check: PASS\n" : "gradient check: FAIL\n");
        } else {
            std::cout << "-- forward\n";
            ok &= check_forward(dir, init, train, test);
            std::cout << "-- training vs sklearn\n";
            ok &= check_training(dir, init, train, lr, check_epochs);
            std::cout << (ok ? "checks: PASS\n" : "checks: FAIL\n");
        }
        return ok ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
