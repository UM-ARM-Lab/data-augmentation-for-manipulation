#include <ompl/base/SpaceInformation.h>
#include <ompl/base/spaces/SE3StateSpace.h>
#include <ompl/base/samplers/ObstacleBasedValidStateSampler.h>
#include <ompl/geometric/SimpleSetup.h>

#include <ompl/config.h>
#include <iostream>
#include <thread>

#include <link_bot_planning/floating_rope_ompl.h>

namespace ob = ompl::base;
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
