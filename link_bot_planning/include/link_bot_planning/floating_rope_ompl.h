#pragma once

#include <ompl/base/goals/GoalSampleableRegion.h>
#include <ompl/control/SpaceInformation.h>

namespace ob = ompl::base;
namespace oc = ompl::control;
namespace og = ompl::geometric;

class MySamplableGoalRegion : public ob::GoalSampleableRegion {
public:
  MySamplableGoalRegion(const oc::SpaceInformationPtr &si);

  bool isSatisfied(const ob::State *st) const override;
};