#include <geometry_msgs/Point.h>
#include <peter_msgs/ComputeOccupancy.h>
#include <ros/callback_queue.h>
#include <ros/ros.h>
#include <ros/subscribe_options.h>
#include <sdf_tools/SDF.h>

#include <arc_utilities/arc_helpers.hpp>
#include <arc_utilities/voxel_grid.hpp>
#include <boost/shared_ptr.hpp>
#include <gazebo/common/common.hh>
#include <gazebo/gazebo.hh>
#include <gazebo/msgs/msgs.hh>
#include <gazebo/physics/ode/ODECollision.hh>
#include <gazebo/physics/ode/ODEPhysics.hh>
#include <gazebo/physics/ode/ODERayShape.hh>
#include <gazebo/physics/ode/ODESphereShape.hh>
#include <gazebo/physics/ode/ODETypes.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/transport/transport.hh>
#include <ignition/math/Vector3.hh>
#include <iostream>
#include <sdf/sdf.hh>
#include <sdf_tools/collision_map.hpp>

namespace gazebo {

class CollisionMapPlugin : public WorldPlugin {
  std::unique_ptr<ros::NodeHandle> ros_node_;
  ros::ServiceServer get_occupancy_service_;
  ros::Publisher sphere_marker_pub_;
  ros::CallbackQueue queue_;
  std::thread ros_queue_thread_;
  std::thread move_sphere_thread_;
  std::thread slow_periodic_thread_;
  physics::PhysicsEnginePtr engine_;
  physics::WorldPtr world_;
  physics::ODEPhysicsPtr ode_;
  physics::ModelPtr m_;
  physics::ODECollisionPtr ode_collision_;

  sdf_tools::CollisionMapGrid grid_;

  bool debug_ = false;
  bool done_ = false;
  double radius_ = 0.005;

  static const sdf_tools::COLLISION_CELL oob_value;
  static const sdf_tools::COLLISION_CELL occupied_value;
  static const sdf_tools::COLLISION_CELL unoccupied_value;

 public:
  void Load(physics::WorldPtr world, sdf::ElementPtr _sdf) override;

 public:
  ~CollisionMapPlugin() override;

 private:
  void QueueThread();

  void SetRadius(physics::ModelPtr m);

  geometry_msgs::Point compute_occupancy_grid(int64_t h_rows, int64_t w_cols, int64_t c_channels,
                                              geometry_msgs::Point center, float resolution,
                                              const std::vector<std::string>& excluded_models);
};

}  // namespace gazebo

void nearCallback(void* _data, dGeomID _o1, dGeomID _o2);

struct MyIntersection {
  std::string name;
  bool in_collision{false};
};
