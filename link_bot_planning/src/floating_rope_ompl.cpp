#include <ompl/base/SpaceInformation.h>
#include <ompl/base/spaces/SE3StateSpace.h>
#include <ompl/base/samplers/ObstacleBasedValidStateSampler.h>
#include <ompl/geometric/SimpleSetup.h>

#include <ompl/config.h>
#include <iostream>
#include <thread>

#include <link_bot_planning/floating_rope_ompl.h>

namespace ob = ompl::base;
namespace oc = ompl::control;
namespace og = ompl::geometric;



class MyValidStateSampler : public ob::ValidStateSampler
{
public:
  MyValidStateSampler(const ob::SpaceInformation *si) : ValidStateSampler(si)
  {
    name_ = "my sampler";
  }

  bool sample(ob::State *state) override
  {
    double* val = static_cast<ob::RealVectorStateSpace::StateType*>(state)->values;
    double z = rng_.uniformReal(-1,1);

    if (z>.25 && z<.5)
    {
      double x = rng_.uniformReal(0,1.8), y = rng_.uniformReal(0,.2);
      switch(rng_.uniformInt(0,3))
      {
      case 0: val[0]=x-1;  val[1]=y-1;  break;
      case 1: val[0]=x-.8; val[1]=y+.8; break;
      case 2: val[0]=y-1;  val[1]=x-1;  break;
      case 3: val[0]=y+.8; val[1]=x-.8; break;
      }
    }
    else
    {
      val[0] = rng_.uniformReal(-1,1);
      val[1] = rng_.uniformReal(-1,1);
    }
    val[2] = z;
    assert(si_->isValid(state));
    return true;
  }

  bool sampleNear(ob::State* /*state*/, const ob::State* /*near*/, const double /*distance*/) override
  {
    throw ompl::Exception("MyValidStateSampler::sampleNear", "not implemented");
    return false;
  }
protected:
  ompl::RNG rng_;
};


bool isStateValid(const ob::State *state)
{
  const ob::RealVectorStateSpace::StateType& pos = *state->as<ob::RealVectorStateSpace::StateType>();
  std::this_thread::sleep_for(ompl::time::seconds(.0005));
  return !(fabs(pos[0])<.8 && fabs(pos[1])<.8 && pos[2]>.25 && pos[2]<.5);
}

ob::ValidStateSamplerPtr allocOBValidStateSampler(const ob::SpaceInformation *si)
{
  return std::make_shared<ob::ObstacleBasedValidStateSampler>(si);
}

ob::ValidStateSamplerPtr allocMyValidStateSampler(const ob::SpaceInformation *si)
{
  return std::make_shared<MyValidStateSampler>(si);
}


class MySamplableGoalRegion : public ob::GoalSampleableRegion
{

  MySamplableGoalRegion(const oc::SpaceInformationPtr &si)
   : GoalSampleableRegion(si)
  {

  }

  bool isSatisfied(const State *st) const override;
  def isSatisfied(self, state: ob.CompoundState, distance):
  state_np = self.scenario_ompl.ompl_state_to_numpy(state)
  rope_points = np.reshape(state_np['rope'], [-1, 3])
  n_from_ends = 7
  near_center_rope_points = rope_points[n_from_ends:-n_from_ends]

  left_gripper_extent = np.reshape(self.goal['left_gripper_box'], [3, 2])
  left_gripper_satisfied = np.logical_and(
      state_np['left_gripper'] >= left_gripper_extent[:, 0],
  state_np['left_gripper'] <= left_gripper_extent[:, 1])

  right_gripper_extent = np.reshape(self.goal['right_gripper_box'], [3, 2])
  right_gripper_satisfied = np.logical_and(
      state_np['right_gripper'] >= right_gripper_extent[:, 0],
  state_np['right_gripper'] <= right_gripper_extent[:, 1])

  point_extent = np.reshape(self.goal['point_box'], [3, 2])
  points_satisfied = np.logical_and(near_center_rope_points >=
                                    point_extent[:, 0], near_center_rope_points <= point_extent[:, 1])
  any_point_satisfied = np.reduce_any(points_satisfied)

  return float(any_point_satisfied and left_gripper_satisfied and right_gripper_satisfied)

  def sampleGoal(self, state_out: ob.CompoundState):
  self.sps.just_sampled_goal = True

# attempt to sample "legit" rope states
  kd = 0.05
  rope = sample_rope_and_grippers(
      self.rng, self.goal['left_gripper'], self.goal['right_gripper'], self.goal['point'],
      FloatingRopeScenario.n_links,
      kd)

  goal_state_np = {
      'left_gripper':  self.goal['left_gripper'],
      'right_gripper': self.goal['right_gripper'],
      'rope':          rope.flatten(),
      'num_diverged':  np.zeros(1, dtype=np.float64),
      'stdev':         np.zeros(1, dtype=np.float64),
  }

  self.scenario_ompl.numpy_to_ompl_state(goal_state_np, state_out)

  if self.plot:
      self.scenario_ompl.s.plot_sampled_goal_state(goal_state_np)

      def distanceGoal(self, state: ob.CompoundState):
  state_np = self.scenario_ompl.ompl_state_to_numpy(state)
  distance = float(self.scenario_ompl.s.distance_grippers_and_any_point_goal(state_np, self.goal).numpy())

# this ensures the goal must have num_diverged = 0
  if state_np['num_diverged'] > 0:
  distance = 1e9
  return distance

      def maxSampleCount(self):
  return 1000

};
