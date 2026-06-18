#include "mjpc/tasks/musculoskeletal/hand_bottle/hand.h"
#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/utilities.h"

namespace mjpc {
  namespace musculoskeletal {
std::string HandBottle::XmlPath() const {
  return GetModelPath("musculoskeletal/hand_bottle/task.xml");
}
std::string HandBottle::Name() const { return "MS HandBottle"; }

void HandBottle::ResidualFn::Residual(const mjModel *model, const mjData *data,
                                   double *residual) const {
  int counter = 0;
  double *bottle_position = SensorByName(model, data, "bottle_position");
  double *goal_position = SensorByName(model, data, "bottle_goal_position");

  mju_sub3(residual + counter, bottle_position, goal_position);
  counter += 3;

  // ---------- Bottle orientation ----------
  double *bottle_orientation = SensorByName(model, data, "bottle_orientation");
  double *goal_bottle_orientation =
      SensorByName(model, data, "bottle_goal_orientation");
  mju_normalize4(goal_bottle_orientation);

  mju_subQuat(residual + counter, goal_bottle_orientation, bottle_orientation);
  counter += 3;

  // ---------- Cube on hand position ----------
  double *hand_position = SensorByName(model, data, "hand_position");
  double *finger1_position = SensorByName(model, data, "finger1_position");
  double *finger2_position = SensorByName(model, data, "finger2_position");
  double *finger3_position = SensorByName(model, data, "finger3_position");
  double *finger4_position = SensorByName(model, data, "finger4_position");
  double *finger5_position = SensorByName(model, data, "finger5_position");

  mju_sub3(residual + counter, hand_position, bottle_position);

  counter += 3;
  mju_sub3(residual + counter, finger1_position, bottle_position);
  counter += 3;
  mju_sub3(residual + counter, finger2_position, bottle_position);
  counter += 3;
  mju_sub3(residual + counter, finger3_position, bottle_position);
  counter += 3;
  mju_sub3(residual + counter, finger4_position, bottle_position);
  counter += 3;
  mju_sub3(residual + counter, finger5_position, bottle_position);
  counter += 3;

  // Sanity check
  CheckSensorDim(model, counter);
}

void HandBottle::TransitionLocked(mjModel *model, mjData *data) {
  // Check for contact between the bottle and the floor
  int bottle = mj_name2id(model, mjOBJ_GEOM, "waterbottle");
  int floor = mj_name2id(model, mjOBJ_GEOM, "floor");

  bool on_floor = false;
  for (int i = 0; i < data->ncon; i++) {
    mjContact *g = data->contact + i;
    if ((g->geom1 == bottle && g->geom2 == floor) || (g->geom2 == bottle && g->geom1 == floor)) {
      on_floor = true;
      break;
    }
  }

  // If the Bottle is on the floor and not moving, reset it
  if (data->time == 0) {
    mju_copy(data->qpos , model->key_qpos, model->nq);
    mju_zero(data->qvel, model->nv);

    // Step the simulation forward
    mutex_.unlock();
    mj_forward(model, data);
    mutex_.lock();
  }  
}
}  // namespace musculoskeletal
}  // namespace mjpcZ