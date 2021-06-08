#include "collision_map_plugin.h"

#include <std_msgs/ColorRGBA.h>
#include <std_msgs/MultiArrayDimension.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <algorithm>
#include <arc_utilities/arc_helpers.hpp>
#include <arc_utilities/ros_helpers.hpp>
#include <arc_utilities/zlib_helpers.hpp>
#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <regex>

using namespace gazebo;

const sdf_tools::COLLISION_CELL CollisionMapPlugin::oob_value{0};
const sdf_tools::COLLISION_CELL CollisionMapPlugin::occupied_value{1};
const sdf_tools::COLLISION_CELL CollisionMapPlugin::unoccupied_value{0};

constexpr auto const PLUGIN_NAME = "collision_map_plugin";

/**
 * This plugin moves a sphere along a grid in the world and checks for collision using raw ODE functions
 * TODOS:
 *  1. set the sphere radius to match resolution
 *  1. make it faster
 */

/** Check if the string s2 matches up until the first double colon **/
bool matches(std::string s1, std::string s2) {
  std::regex self_regex(s2 + "::.*");
  return std::regex_search(s1, self_regex);
}

void CollisionMapPlugin::Load(physics::WorldPtr world, sdf::ElementPtr /*sdf*/) {
  world_ = world;
  engine_ = world->Physics();
  engine_->InitForThread();

  ode_ = boost::dynamic_pointer_cast<physics::ODEPhysics>(engine_);

  world_->InsertModelFile("model://collision_sphere");

  if (!ros::isInitialized()) {
    ROS_FATAL_STREAM("A ROS node for Gazebo has not been initialized, unable to load plugin. "
                     << "Load the Gazebo system plugin 'libgazebo_ros_api_plugin.so' in the gazebo_ros package)");
    return;
  }

  auto get_occupancy = [&](peter_msgs::ComputeOccupancyRequest &req, peter_msgs::ComputeOccupancyResponse &res) {
    auto const &origin_point =
        compute_occupancy_grid(req.h_rows, req.w_cols, req.c_channels, req.center, req.resolution, req.excluded_models);

    auto const grid_float = [&]() {
      auto const &data = grid_.GetImmutableRawData();
      std::vector<float> flat;
      for (auto const &d : data) {
        flat.emplace_back(d.occupancy);
      }
      return flat;
    }();
    res.grid = grid_float;
    res.origin_point = origin_point;
    std_msgs::MultiArrayDimension row_dim;
    row_dim.label = "row";
    row_dim.size = grid_.GetNumXCells();
    row_dim.stride = 1;
    std_msgs::MultiArrayDimension col_dim;
    col_dim.label = "col";
    col_dim.size = grid_.GetNumYCells();
    col_dim.stride = 1;
    std_msgs::MultiArrayDimension channel_dim;
    channel_dim.label = "channel";
    channel_dim.size = grid_.GetNumZCells();
    channel_dim.stride = 1;
    return true;
  };

  ros_node_ = std::make_unique<ros::NodeHandle>("collision_map_plugin");

  {
    auto so = ros::AdvertiseServiceOptions::create<peter_msgs::ComputeOccupancy>("/occupancy", get_occupancy,
                                                                                 ros::VoidConstPtr(), &queue_);
    get_occupancy_service_ = ros_node_->advertiseService(so);

    sphere_marker_pub_ = ros_node_->advertise<visualization_msgs::MarkerArray>("collision_map_grid", 10, false);

    debug_ = ROSHelpers::GetParam<bool>(*ros_node_, "debug", false);
  }

  ros_queue_thread_ = std::thread([this] { QueueThread(); });

  move_sphere_thread_ = std::thread([this]() {
    // wait for the model to exist
    ROS_INFO_NAMED(PLUGIN_NAME, "waiting for collision_sphere to appear...");
    while (true) {
      physics::ModelPtr m = world_->ModelByName("collision_sphere");
      if (m) {
        boost::recursive_mutex::scoped_lock lock(*engine_->GetPhysicsUpdateMutex());
        // no way we will be in the way of anything 10 meters underground...
        m->SetWorldPose({0, 0, -10, 0, 0, 0});

        SetRadius(m);
        break;
      }
    }
    ROS_INFO_NAMED(PLUGIN_NAME, "done waiting for collision_sphere");
  });

  slow_periodic_thread_ = std::thread([this] {
    while (not done_) {
      sleep(10);
      debug_ = ROSHelpers::GetParamDebugLog<bool>(*ros_node_, "debug", false);
    }
  });
}

void CollisionMapPlugin::SetRadius(physics::ModelPtr m) {
  auto const link = m->GetLink("link_1");
  if (!link) {
    ROS_ERROR_STREAM_NAMED(PLUGIN_NAME, "Could not find link");
    return;
  }

  auto const collision = link->GetCollision("collision");
  if (!collision) {
    ROS_ERROR_STREAM_NAMED(PLUGIN_NAME, "Could not find collision");
    return;
  }

  auto const shape = collision->GetShape();
  auto const sphere = boost::dynamic_pointer_cast<physics::SphereShape>(shape);
  if (!sphere) {
    ROS_ERROR_STREAM_NAMED(PLUGIN_NAME, "Could not cast to sphere");
    return;
  }

  radius_ = sphere->GetRadius();
  ROS_DEBUG_STREAM_NAMED(PLUGIN_NAME, "Set radius to " << radius_);
}

geometry_msgs::Point CollisionMapPlugin::compute_occupancy_grid(int64_t h_rows, int64_t w_cols, int64_t c_channels,
                                                                geometry_msgs::Point center, float resolution,
                                                                std::vector<std::string> const &excluded_models) {
  auto const x_width = resolution * w_cols;
  auto const y_height = resolution * h_rows;
  auto const z_size = resolution * c_channels;
  Eigen::Isometry3d origin_transform = Eigen::Isometry3d::Identity();
  origin_transform.translation() =
      Eigen::Vector3d{center.x - x_width / 2.f, center.y - y_height / 2.f, center.z - z_size / 2.f};

  grid_ = sdf_tools::CollisionMapGrid(origin_transform, "/world", resolution, w_cols, h_rows, c_channels, oob_value);
  ROS_DEBUG_STREAM_NAMED(PLUGIN_NAME, "origin " << origin_transform.translation() << " shape [" << h_rows << ","
                                                << w_cols << "," << c_channels << "]");

  auto const t0 = std::chrono::steady_clock::now();

  m_ = world_->ModelByName("collision_sphere");
  if (!m_) {
    ROS_WARN_STREAM_NAMED(PLUGIN_NAME, "Collision sphere is not in the world (yet)");
    geometry_msgs::Point invalid;
    invalid.x = -999;
    invalid.y = -999;
    invalid.z = -999;
    return invalid;
  }

  auto c = m_->GetChildCollision("collision");
  ode_collision_ = boost::dynamic_pointer_cast<physics::ODECollision>(c);
  auto const sphere_collision_geom_id = ode_collision_->GetCollisionId();

  visualization_msgs::MarkerArray markers;
  visualization_msgs::Marker marker;
  marker.type = visualization_msgs::Marker::SPHERE;
  marker.action = visualization_msgs::Marker::ADD;
  marker.scale.x = radius_ * 2;
  marker.scale.y = radius_ * 2;
  marker.scale.z = radius_ * 2;
  marker.pose.orientation.w = 1;
  marker.header.stamp = ros::Time::now();
  marker.header.frame_id = "world";

  // lock physics engine while creating/testing collision. not sure this is necessary.
  {
    boost::recursive_mutex::scoped_lock lock(*engine_->GetPhysicsUpdateMutex());
    for (auto x_idx{0l}; x_idx < grid_.GetNumXCells(); ++x_idx) {
      for (auto y_idx{0l}; y_idx < grid_.GetNumYCells(); ++y_idx) {
        for (auto z_idx{0l}; z_idx < grid_.GetNumZCells(); ++z_idx) {
          auto const grid_location = grid_.GridIndexToLocation(x_idx, y_idx, z_idx);
          m_->SetWorldPose({grid_location[0], grid_location[1], grid_location[2], 0, 0, 0});
          MyIntersection intersection;
          auto const collision_space = (dGeomID)(ode_->GetSpaceId());
          dSpaceCollide2(sphere_collision_geom_id, collision_space, &intersection, &nearCallback);
          if (intersection.in_collision) {
            bool exclude = std::any_of(
                excluded_models.cbegin(), excluded_models.cend(),
                [intersection](auto const &excluded_model) { return matches(intersection.name, excluded_model); });

            if (not exclude) {
              if (debug_) {
                marker.id = x_idx * grid_.GetNumYCells() * grid_.GetNumZCells() + y_idx * grid_.GetNumZCells() + z_idx;
                marker.pose.position.x = grid_location[0];
                marker.pose.position.y = grid_location[1];
                marker.pose.position.z = grid_location[2];
                marker.color.r = 1;
                marker.color.g = 0;
                marker.color.a = 1;
                markers.markers.push_back(marker);
                ROS_DEBUG_STREAM_NAMED(PLUGIN_NAME + ".collision", "collision with " << intersection.name);
              }
              grid_.SetValue(x_idx, y_idx, z_idx, occupied_value);
            } else {
              if (debug_) {
                ROS_DEBUG_STREAM_NAMED(PLUGIN_NAME + ".collision", "excluding collision with " << intersection.name);
              }
              grid_.SetValue(x_idx, y_idx, z_idx, unoccupied_value);
            }
          } else if (debug_) {
            marker.id = x_idx * grid_.GetNumYCells() * grid_.GetNumZCells() + y_idx * grid_.GetNumZCells() + z_idx;
            marker.pose.position.x = grid_location[0];
            marker.pose.position.y = grid_location[1];
            marker.pose.position.z = grid_location[2];
            marker.color.r = 0;
            marker.color.g = 1;
            marker.color.a = 0.1;
            markers.markers.push_back(marker);
          }
        }
      }
    }
  }

  if (debug_) {
    sphere_marker_pub_.publish(markers);
  }

  auto const t1 = std::chrono::steady_clock::now();
  std::chrono::duration<double> const time_to_compute_occupancy_grid = t1 - t0;
  ROS_DEBUG_STREAM_NAMED(PLUGIN_NAME + ".perf",
                         "Time to compute occupancy grid: " << time_to_compute_occupancy_grid.count());

  geometry_msgs::Point origin_point;
  auto const &origin_location = grid_.GridIndexToLocation(0, 0, 0);
  origin_point.x = origin_location.x();
  origin_point.y = origin_location.y();
  origin_point.z = origin_location.z();
  return origin_point;
}

CollisionMapPlugin::~CollisionMapPlugin() {
  done_ = true;
  queue_.clear();
  queue_.disable();
  ros_node_->shutdown();
  ros_queue_thread_.join();
  slow_periodic_thread_.join();
}

void CollisionMapPlugin::QueueThread() {
  double constexpr timeout = 0.01;
  while (ros_node_->ok()) {
    queue_.callAvailable(ros::WallDuration(timeout));
  }
}

void nearCallback(void *_data, dGeomID _o1, dGeomID _o2) {
  auto intersection = static_cast<MyIntersection *>(_data);

  if (dGeomIsSpace(_o2)) {
    dSpaceCollide2(_o1, _o2, _data, &nearCallback);
  } else {
    auto const ode_collision = static_cast<physics::ODECollision *>(dGeomGetData(_o2));
    if (ode_collision) {
      dContactGeom contact;
      if (dGeomGetClass(_o2) == dTriMeshClass) {
        ROS_DEBUG_STREAM_NAMED(PLUGIN_NAME + ".collision",
                               "Skipping unimplemented collision with mesh " << ode_collision->GetScopedName().c_str());
      } else {
        int n = dCollide(_o1, _o2, 1, &contact, sizeof(contact));
        if (n > 0) {
          if (intersection) {
            intersection->name = ode_collision->GetScopedName();
            intersection->in_collision = true;
          }
        }
      }
    } else {
      intersection->in_collision = false;
    }
  }
}

// Register this plugin with the simulator
GZ_REGISTER_WORLD_PLUGIN(CollisionMapPlugin)
