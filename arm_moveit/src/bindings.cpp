#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <moveit/python/pybind_rosmsg_typecasters.h>

#include <arm_moveit/robot_points_generator.h>

namespace py = pybind11;

PYBIND11_MODULE(pyrobot_points_generator, m) {
  py::class_<RobotPointsGenerator>(m, "RobotPointsGenerator")
      .def(py::init<double>(), py::arg("res"))
      .def("get_link_names", &RobotPointsGenerator::getLinkModelNames)
      .def("check_collision", &RobotPointsGenerator::checkCollision)
      .def("get_robot_name", &RobotPointsGenerator::getRobotName)
      //
      ;
}
