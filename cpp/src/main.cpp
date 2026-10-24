#include <Eigen/Dense>
#include <iostream>

int main() {
    std::cout << "Eigen version: " << EIGEN_WORLD_VERSION << "."
              << EIGEN_MAJOR_VERSION << "." << EIGEN_MINOR_VERSION << "\n";

    Eigen::MatrixXd a(2, 3);
    a << 1, 2, 3,
         4, 5, 6;
    Eigen::MatrixXd b(3, 2);
    b << 7, 8,
         9, 10,
         11, 12;

    std::cout << "A * B =\n" << a * b << "\n";
    return 0;
}
