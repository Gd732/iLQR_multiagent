#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>

namespace py = pybind11;

py::array_t<double> compute_B_matrix(double Ix, double Iy, double Iz, double m, double dt, 
                                     double tx, double ty, double tz, int horizon) {
    py::array_t<double> B = py::array_t<double>({12, 4, horizon});
    auto B_mutable = B.mutable_unchecked<3>();

    for (int t = 0; t < horizon; ++t) {
        B_mutable(3, 0, t) = (dt * std::sin(ty) * std::sin(tz)) / m;
        B_mutable(4, 0, t) = -(dt * std::cos(tz) * std::sin(ty)) / m;
        B_mutable(5, 0, t) = (dt * std::cos(ty)) / m;
        B_mutable(6, 1, t) = dt * dt / Ix;
        B_mutable(6, 2, t) = (dt * dt * std::sin(tx) * std::sin(ty)) / (Iy * std::cos(ty));
        B_mutable(6, 3, t) = (dt * dt * std::cos(tx) * std::sin(ty)) / (Iz * std::cos(ty));
        B_mutable(7, 2, t) = (dt * dt * std::cos(tx)) / Iy;
        B_mutable(7, 3, t) = -(dt * dt * std::sin(tx)) / Iz;
        B_mutable(8, 2, t) = (dt * dt * std::sin(tx)) / (Iy * std::cos(ty));
        B_mutable(8, 3, t) = (dt * dt * std::cos(tx)) / (Iz * std::cos(ty));
        B_mutable(9, 1, t) = dt / Ix;
        B_mutable(9, 2, t) = (dt * std::sin(tx) * std::sin(ty)) / (Iy * std::cos(ty));
        B_mutable(9, 3, t) = (dt * std::cos(tx) * std::sin(ty)) / (Iz * std::cos(ty));
        B_mutable(10, 2, t) = (dt * std::cos(tx)) / Iy;
        B_mutable(10, 3, t) = -(dt * std::sin(tx)) / Iz;
        B_mutable(11, 2, t) = (dt * std::sin(tx)) / (Iy * std::cos(ty));
        B_mutable(11, 3, t) = (dt * std::cos(tx)) / (Iz * std::cos(ty));
    }

    return B;
}

PYBIND11_MODULE(compute_B_matrix, m) {
    m.def("compute_B_matrix", &compute_B_matrix, "Compute the B matrix");
}
