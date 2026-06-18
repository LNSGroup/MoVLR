#include "mjpc/tasks/musculoskeletal/slope/walk.h"

#include <string>

#include <absl/random/random.h>
#include <mujoco/mujoco.h>
#include "mjpc/utilities.h"

namespace mjpc::musculoskeletal {

std::string Slope::XmlPath() const {
  return GetModelPath("musculoskeletal/slope/task.xml");
}
std::string Slope::Name() const { return "MS Slope"; }
// ------------------ Residuals for MS humanoid stand task ------------
//   Number of residuals: 6
//     Residual (0): Desired height
//     Residual (1): Balance: COM_xy - average(feet position)_xy
//     Residual (2): Com Vel: should be 0 and equal feet average vel
//     Residual (3): Control: minimise control
//     Residual (4): Joint vel: minimise joint velocity
//     Residual (5): Trunk Stability: minimize deviation from upright posture
//     Residual (6): Forward Velocity: achieve and maintain target velocity
//     Residual (7): Foot Placement: ensure feet are placed correctly
//   Number of parameters: 1
//     Parameter (0): height_goal
// -------------------------------------------------------------
void Slope::ResidualFn::Residual(const mjModel* model, const mjData* data,
                       double* residual) const {

  int counter = 0;
  // ----- Height: head feet vertical error ----- //
  // feet sensor positions
  double* f1_position = SensorByName(model, data, "sp0");
  double* f2_position = SensorByName(model, data, "sp1");
  double* f3_position = SensorByName(model, data, "sp2");
  double* f4_position = SensorByName(model, data, "sp3");
  double* head_position = SensorByName(model, data, "head_position");
  double feet_height = 0;
  double fxy_avg[2] = {0, 0};
  int on_groud_cnt = 0;

  if (on_groud_cnt == 0) {
    fxy_avg[0] = 0.25 * (f1_position[0] + f2_position[0] + f3_position[0] + f4_position[0]);
    fxy_avg[1] = 0.25 * (f1_position[1] + f2_position[1] + f3_position[1] + f4_position[1]);
  }
  else {
    fxy_avg[0] /= on_groud_cnt;
    fxy_avg[1] /= on_groud_cnt;
  }
  
  feet_height = 0.25 * (f1_position[2] + f2_position[2] + f3_position[2] + f4_position[2]);
  double head_feet_error = head_position[2] - feet_height;
  residual[counter++] = mju_min(head_feet_error - parameters_[0], 0);
  
  // ----- Balance: COM_xy - average(feet position)_xy ----- //
  double* com_position = SensorByName(model, data, "pelvis_subtreecom");
  double* com_velocity = SensorByName(model, data, "pelvis_subtreelinvel");
  double capture_point[3] = {com_position[0], com_position[1], com_position[2]};
  mju_subFrom(fxy_avg, capture_point, 2);
  double com_feet_distance = mju_norm(fxy_avg, 2);
  residual[counter++] = com_feet_distance;

  // ----- Com Vel: should be 0 and equal feet average vel ----- //
  double* foot_right_vel = SensorByName(model, data, "foot_right_vel");
  double* foot_left_vel = SensorByName(model, data, "foot_left_vel");
  double avg_foot_vel[3] = {0, 0, 0};
  mju_add(avg_foot_vel, foot_right_vel, foot_left_vel, 3);
  mju_scl(avg_foot_vel, avg_foot_vel, 0.5, 3);
  double com_vel_error[3] = {0, 0, 0};
  mju_sub(com_vel_error, com_velocity, avg_foot_vel, 3);
  residual[counter++] = mju_norm(com_vel_error, 3);

  // ----- Control: minimise control ----- //
  residual[counter++] = mju_norm(data->ctrl, model->nu);

  // ----- Joint vel: minimise joint velocity ----- //
  residual[counter++] = mju_norm(data->qvel+6, 85-6);

  // ----- Trunk Stability: minimize deviation from upright posture ----- //
  double* torso_up = SensorByName(model, data, "torso_up");
  double upright[3] = {0, 1, 0};
  double deviation = 1 - mju_dot3(torso_up, upright);
  residual[counter++] = deviation;

  // ----- Forward Velocity: achieve and maintain target velocity ----- //
  double target_velocity = 1.0;
  double forward_velocity_error = com_velocity[0] - target_velocity;
  residual[counter++] = forward_velocity_error;

  // ----- Foot Placement: ensure feet are placed correctly ----- //
  double foot_right_position[3] = {0, 0, 0};
  double foot_left_position[3] = {0, 0, 0};
  mju_copy(foot_right_position, f2_position, 3);
  mju_copy(foot_left_position, f4_position, 3);
  double foot_placement_error[3] = {0, 0, 0};
  mju_sub(foot_placement_error, foot_right_position, foot_left_position, 3);
  residual[counter++] = mju_norm(foot_placement_error, 3);

  int user_sensor_dim = 0;
  for (int i = 0; i < model->nsensor; i++) {
    if (model->sensor_type[i] == mjSENS_USER) {
      user_sensor_dim += model->sensor_dim[i];
    }
  }

  if (user_sensor_dim != counter) {
    std::cout<<"sensor dim"<<user_sensor_dim<<" "<<counter<<std::endl;
    mju_error_i(
        "mismatch between total user-sensor dimension "
        "and actual length of residual %d %d", counter);
  }
}

void Slope::TransitionLocked(mjModel* model, mjData* data) {
  if (data->time == 0) {
    mju_copy(data->qpos, model->key_qpos, model->nq);
    mju_zero(data->qvel, model->nv);
    mju_zero(data->act, model->na);
  }
}
}