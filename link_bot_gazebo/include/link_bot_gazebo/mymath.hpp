#pragma once

#include <ignition/math.hh>

// FIXME: move these to arc_utilities

/** \brief Computes the signed shorted angle between y2 and y1. Check CommonTest.cpp to see examples
 *
 * @param y1 the second angle in the subtraction
 * @param y2 the first angle in the subtraction
 * @return the signed shorted angle between y2 and y1.
 */
constexpr double angle_error(double const y1, double const y2) {
  double diff = y2 - y1;
  if (diff > M_PI) return diff - M_PI * 2;
  if (diff < -M_PI) return diff + M_PI * 2;
  return diff;
}

ignition::math::Vector3d angle_error(ignition::math::Vector3d const y1, ignition::math::Vector3d const y2);

inline double quat_diff(ignition::math::Quaterniond q1, ignition::math::Quaterniond q2) {
  //  http://www.cs.cmu.edu/~cga/dynopt/readings/Rmetric.pdf, see equation (20)
  auto const dot = q1.Dot(q2);
  return 1 - abs(dot);
}
