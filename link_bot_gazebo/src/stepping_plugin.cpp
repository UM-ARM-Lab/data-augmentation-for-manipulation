#include <peter_msgs/WorldControl.h>
#include <ros/callback_queue.h>
#include <ros/ros.h>
#include <ros/subscribe_options.h>

#include <atomic>
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <ignition/math/Pose3.hh>
#include <memory>

// this is a free function because
// (1) to remove a clang-tidy "complexity" warning
// (2) because the string is long and doesn't look good when 3 levels on indent int
void print_ros_init_error()
{
  ROS_FATAL("A ROS node for Gazebo has not been initialized, unable to load plugin. "
            "Load the Gazebo system plugin 'libgazebo_ros_api_plugin.so' in the gazebo_ros package");
}


namespace gazebo
{
class SteppingPlugin : public WorldPlugin
{
 public:
  void Load(physics::WorldPtr parent, sdf::ElementPtr /*_sdf*/) override
  {
    if (!ros::isInitialized())
    {
      print_ros_init_error();
      return;
    }
    seconds_per_step_ = parent->Physics()->GetMaxStepSize();

    ros_node_ = std::make_unique<ros::NodeHandle>("stepping_plugin");
    service_ = ros_node_->advertiseService("/world_control", &SteppingPlugin::onWorldControl, this);

    async_spinner_ = std::make_unique<ros::AsyncSpinner>(0);
    async_spinner_->start();

    transport::NodePtr node(new transport::Node());
    node->Init(parent->Name());
    pub_ = node->Advertise<msgs::WorldControl>("~/world_control");

    step_connection__ = event::Events::ConnectWorldUpdateEnd([&]()
                                                             {
                                                               if (step_count_ > 0)
                                                               { --step_count_; }
                                                             });

    ROS_INFO("Finished loading stepping plugin!");
  }

  bool onWorldControl(peter_msgs::WorldControlRequest &req, peter_msgs::WorldControlResponse &res)
  {
    (void) res;
    auto const steps = [this, req]()
    {
      if (req.seconds > 0)
      {
        return static_cast<unsigned int>(req.seconds / seconds_per_step_);
      } else
      {
        return req.steps;
      }
    }();
    step_count_ = steps;
    msgs::WorldControl gz_msg;
    gz_msg.set_multi_step(steps);
    pub_->Publish(gz_msg);
    while (step_count_ != 0);
    return true;
  }

  ~SteppingPlugin() override
  {
    async_spinner_->stop();
  }

  transport::PublisherPtr pub_;
  event::ConnectionPtr step_connection__;
  std::unique_ptr<ros::NodeHandle> ros_node_;
  ros::ServiceServer service_;
  std::unique_ptr<ros::AsyncSpinner> async_spinner_;
  std::atomic<int> step_count_{0};
  double seconds_per_step_{0.0};

};

// Register this plugin with the simulator
GZ_REGISTER_WORLD_PLUGIN(SteppingPlugin)
}  // namespace gazebo
