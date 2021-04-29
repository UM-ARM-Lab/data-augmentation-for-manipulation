#include "voxel_visual.h"

#include <OGRE/OgreSceneManager.h>
#include <OGRE/OgreSceneNode.h>
#include <OGRE/OgreVector3.h>
#include <rviz/ogre_helpers/point_cloud.h>
#include <std_msgs/Float32MultiArray.h>

namespace mps_shape_completion_visualization {

// BEGIN_TUTORIAL
VoxelGridVisual::VoxelGridVisual(Ogre::SceneManager *scene_manager, Ogre::SceneNode *parent_node) {
  scene_manager_ = scene_manager;

  frame_node_ = parent_node->createChildSceneNode();

  // We create the arrow object within the frame node so that we can
  // set its position and direction relative to its header frame.
  // voxel_grid_.reset(new rviz::PointCloud( scene_manager_, frame_node_ ));
  voxel_grid_.reset(new rviz::PointCloud());
  voxel_grid_->setRenderMode(rviz::PointCloud::RM_BOXES);
  frame_node_->attachObject(voxel_grid_.get());
}

VoxelGridVisual::~VoxelGridVisual() {
  // Destroy the frame node since we don't need it anymore.
  scene_manager_->destroySceneNode(frame_node_);
}

void VoxelGridVisual::setMessage(const mps_shape_completion_msgs::OccupancyStamped::ConstPtr &msg) {
  latest_msg = *msg;
  updatePointCloud();
}

void VoxelGridVisual::updatePointCloud() {
  if (latest_msg.occupancy.layout.dim.size() == 0) {
    return;
  }

  auto const scale = static_cast<float>(latest_msg.scale);
  voxel_grid_->setDimensions(scale, scale, scale);

  auto const data = latest_msg.occupancy.data;
  auto const colors = latest_msg.colors;
  auto const dims = latest_msg.occupancy.layout.dim;
  auto const data_offset = latest_msg.occupancy.layout.data_offset;

  std::vector<rviz::PointCloud::Point> points;
  for (int i = 0; i < dims[0].size; i++) {
    for (int j = 0; j < dims[1].size; j++) {
      for (int k = 0; k < dims[2].size; k++) {
        auto const index = data_offset + dims[1].stride * i + dims[2].stride * j + k;
        auto val = data[index];
        auto const color = [&]() {
          if (not colors.empty()) {
            return colors[index];
          }
          // if the user sets nothing, assume they want to use the color that was set in the rviz color picker
          else if (latest_msg.color.r == 0 and latest_msg.color.g == 0 and latest_msg.color.b == 0 and latest_msg.color.a == 0)
          {
            std_msgs::ColorRGBA color;
            color.r = r_;
            color.g = g_;
            color.b = b_;
            color.a = a_;
            return color;
          }
          return latest_msg.color;
        }();

        if (val < threshold_) {
          continue;
        }

        rviz::PointCloud::Point p;
        p.position.x = scale / 2.f + static_cast<float>(i) * scale;
        p.position.y = scale / 2.f + static_cast<float>(j) * scale;
        p.position.z = scale / 2.f + static_cast<float>(k) * scale;

        if (binary_display_) {
          val = 1.0;
        }

        p.setColor(color.r, color.g, color.b, std::min(val * a_, 1.f));

        points.push_back(p);
      }
    }
  }

  voxel_grid_->clear();

  // The per-point alpha setting is not great with alpha=1, so in
  //  certain cases do not use it
  bool use_per_point = !(a_ >= 1.0 && binary_display_);
  voxel_grid_->setAlpha(a_, use_per_point);

  voxel_grid_->addPoints(&points.front(), points.size());
}

// Position and orientation are passed through to the SceneNode.
void VoxelGridVisual::setFramePosition(const Ogre::Vector3 &position) { frame_node_->setPosition(position); }

void VoxelGridVisual::setFrameOrientation(const Ogre::Quaternion &orientation) {
  frame_node_->setOrientation(orientation);
}

void VoxelGridVisual::setColor(float r, float g, float b, float a) {
  r_ = r;
  g_ = g;
  b_ = b;
  a_ = a;
}

void VoxelGridVisual::setBinaryDisplay(bool binary_display) { binary_display_ = binary_display; }

void VoxelGridVisual::setThreshold(float threshold) { threshold_ = threshold; }

}  // end namespace mps_shape_completion_visualization
