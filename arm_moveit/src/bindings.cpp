#include <arm_moveit/robot_points_generator.h>
#include <moveit/python/pybind_rosmsg_typecasters.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(pyrobot_points_generator, m) {
  py::class_<RobotPointsGenerator>(m, "RobotPointsGenerator")
      .def(py::init<double, std::string const &, double>(), py::arg("res"), py::arg("robot_description"), py::arg("collision_sphere_radius"))
      .def("get_link_names", &RobotPointsGenerator::getLinkModelNames)
      .def("check_collision", &RobotPointsGenerator::checkCollision)
      .def("get_robot_name", &RobotPointsGenerator::getRobotName)
      //
      ;
}
