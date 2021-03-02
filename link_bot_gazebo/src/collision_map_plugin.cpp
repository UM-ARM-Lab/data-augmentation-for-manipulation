#include "collision_map_plugin.h"

#include <algorithm>
#include <std_msgs/ColorRGBA.h>
#include <std_msgs/MultiArrayDimension.h>
#include <visualization_msgs/Marker.h>

#include <arc_utilities/arc_helpers.hpp>
#include <arc_utilities/zlib_helpers.hpp>
#include <atomic>
#include <chrono>
#include <functional>
#include <regex>
#include <memory>

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
bool matches(std::string s1, std::string s2)
{
  std::regex self_regex(s2 + "::.*");
  return std::regex_search(s1, self_regex);
}

void CollisionMapPlugin::Load(physics::WorldPtr world, sdf::ElementPtr /*sdf*/)
{
  world_ = world;
  engine_ = world->Physics();
  engine_->InitForThread();

  ode_ = boost::dynamic_pointer_cast<physics::ODEPhysics>(engine_);

  world_->InsertModelFile("model://collision_sphere");

  if (!ros::isInitialized())
  {
    ROS_FATAL_STREAM("A ROS node for Gazebo has not been initialized, unable to load plugin. "
                         << "Load the Gazebo system plugin 'libgazebo_ros_api_plugin.so' in the gazebo_ros package)");
    return;
  }

  auto get_occupancy = [&](peter_msgs::ComputeOccupancyRequest &req, peter_msgs::ComputeOccupancyResponse &res)
  {
    compute_occupancy_grid(req.h_rows, req.w_cols, req.c_channels, req.center, req.resolution, req.excluded_models);

    auto const grid_float = [&]()
    {
      auto const &data = grid_.GetImmutableRawData();
      std::vector<float> flat;
      for (auto const &d : data)
      {
        flat.emplace_back(d.occupancy);
      }
      return flat;
    }();
    res.grid = grid_float;
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
  }

  ros_queue_thread_ = std::thread([this] { QueueThread(); });

  move_sphere_thread_ = std::thread(
      [this]()
      {
        // wait for the model to exist
        ROS_INFO_NAMED(PLUGIN_NAME, "waiting for collision_sphere to appear...");
        while (true)
        {
          physics::ModelPtr m = world_->ModelByName("collision_sphere");
          if (m)
          {
            boost::recursive_mutex::scoped_lock lock(*engine_->GetPhysicsUpdateMutex());
            // no way we will be in the way of anything 10 meters underground...
            m->SetWorldPose({0, 0, -10, 0, 0, 0});
            break;
          }
        }
        ROS_INFO_NAMED(PLUGIN_NAME, "done waiting for collision_sphere");
      });
}

void CollisionMapPlugin::compute_occupancy_grid(int64_t h_rows, int64_t w_cols, int64_t c_channels,
                                                geometry_msgs::Point center, float resolution,
                                                std::vector<std::string> const &excluded_models)
{
  auto const x_width = resolution * w_cols;
  auto const y_height = resolution * h_rows;
  auto const z_size = resolution * c_channels;
  Eigen::Isometry3d origin_transform = Eigen::Isometry3d::Identity();
  origin_transform.translation() =
      Eigen::Vector3d{center.x - x_width / 2, center.y - y_height / 2, center.z - z_size / 2};

  grid_ = sdf_tools::CollisionMapGrid(origin_transform, "/world", resolution, w_cols, h_rows, c_channels, oob_value);

  auto const t0 = std::chrono::steady_clock::now();

  m_ = world_->ModelByName("collision_sphere");
  if (!m_)
  {
    ROS_WARN_STREAM_NAMED(PLUGIN_NAME, "Collision sphere is not in the world (yet)");
    return;
  }

  auto c = m_->GetChildCollision("collision");
  ode_collision_ = boost::dynamic_pointer_cast<physics::ODECollision>(c);
  auto const sphere_collision_geom_id = ode_collision_->GetCollisionId();

  // lock physics engine while creating/testing collision. not sure this is necessary.
  {
    boost::recursive_mutex::scoped_lock lock(*engine_->GetPhysicsUpdateMutex());

    for (auto x_idx{0l}; x_idx < grid_.GetNumXCells(); ++x_idx)
    {
      for (auto y_idx{0l}; y_idx < grid_.GetNumYCells(); ++y_idx)
      {
        for (auto z_idx{0l}; z_idx < grid_.GetNumZCells(); ++z_idx)
        {
          auto const grid_location = grid_.GridIndexToLocation(x_idx, y_idx, z_idx);
          ROS_DEBUG_STREAM_NAMED(PLUGIN_NAME, "Checking " << grid_location[0] << " " << grid_location[1] << " " << grid_location[2]);
          m_->SetWorldPose({grid_location[0], grid_location[1], grid_location[2], 0, 0, 0});
          MyIntersection intersection;
          auto const collision_space = (dGeomID) (ode_->GetSpaceId());
          dSpaceCollide2(sphere_collision_geom_id, collision_space, &intersection, &nearCallback);
          if (intersection.in_collision)
          {
            bool exclude = std::any_of(excluded_models.cbegin(), excluded_models.cend(),
                                       [intersection](auto const &excluded_model)
                                       {
                                         return matches(intersection.name, excluded_model);
                                       });

            if (not exclude)
            {
              ROS_DEBUG_STREAM_NAMED(PLUGIN_NAME, "collision with " << intersection.name);
              grid_.SetValue(x_idx, y_idx, z_idx, occupied_value);
            } else
            {
              ROS_DEBUG_STREAM_NAMED(PLUGIN_NAME, "excluding collision with " << intersection.name);
              grid_.SetValue(x_idx, y_idx, z_idx, unoccupied_value);
            }
          }
        }
      }
    }
  }

  auto const t1 = std::chrono::steady_clock::now();
  std::chrono::duration<double> const time_to_compute_occupancy_grid = t1 - t0;
  gzlog << "Time to compute occupancy grid: " << time_to_compute_occupancy_grid.count() << std::endl;
}

CollisionMapPlugin::~CollisionMapPlugin()
{
  queue_.clear();
  queue_.disable();
  ros_node_->shutdown();
  ros_queue_thread_.join();
}

void CollisionMapPlugin::QueueThread()
{
  double constexpr timeout = 0.01;
  while (ros_node_->ok())
  {
    queue_.callAvailable(ros::WallDuration(timeout));
  }
}

void nearCallback(void *_data, dGeomID _o1, dGeomID _o2)
{
  auto intersection = static_cast<MyIntersection *>(_data);

  if (dGeomIsSpace(_o2))
  {
    dSpaceCollide2(_o1, _o2, _data, &nearCallback);
  } else
  {
    auto const ode_collision = static_cast<physics::ODECollision *>(dGeomGetData(_o2));
    if (ode_collision)
    {
      dContactGeom contact;
      if (dGeomGetClass(_o2) == dTriMeshClass)
      {
        ROS_DEBUG_STREAM_NAMED(PLUGIN_NAME,
                               "Skipping unimplemented collision with mesh "
                                   << ode_collision->GetScopedName().c_str());
      } else
      {
        int n = dCollide(_o1, _o2, 1, &contact, sizeof(contact));
        if (n > 0)
        {
          if (intersection)
          {
            intersection->name = ode_collision->GetScopedName();
            intersection->in_collision = true;
          }
        }
      }
    } else
    {
      intersection->in_collision = false;
    }
  }
}

// Register this plugin with the simulator
GZ_REGISTER_WORLD_PLUGIN(CollisionMapPlugin)
