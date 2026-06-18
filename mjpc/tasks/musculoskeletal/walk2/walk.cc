// Copyright 2022 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mjpc/tasks/musculoskeletal/walk2/walk.h"

#include <string>

#include <absl/random/random.h>
#include <mujoco/mujoco.h>
#include "mjpc/utilities.h"

namespace mjpc::musculoskeletal {

std::string Walk2::XmlPath() const {
  return GetModelPath("musculoskeletal/walk2/task.xml");
}
std::string Walk2::Name() const { return "MS Walk2"; }
// ------------------ Residuals for MS humanoid stand task ------------
//   Number of residuals: 6
//     Residual (0): Desired height
//     Residual (1): Balance: COM_xy - average(feet position)_xy
//     Residual (2): Com Vel: should be 0 and equal feet average vel
//     Residual (3): Control: minimise control
//     Residual (4): Joint vel: minimise joint velocity
//   Number of parameters: 1
//     Parameter (0): height_goal
// -------------------------------------------------------------
void Walk2::ResidualFn::Residual(const mjModel* model, const mjData* data,
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
  
  feet_height = 0.25 * (f1_position[2] + f2_position[2] + f3_position[2] + f4_position[2]);
  fxy_avg[0] = 0.25 * (f1_position[0] + f2_position[0] + f3_position[0] + f4_position[0]);
  fxy_avg[1] = 0.25 * (f1_position[1] + f2_position[1] + f3_position[1] + f4_position[1]);

  double head_feet_error =
      head_position[2] - feet_height;
  residual[counter++] = head_feet_error - parameters_[0];
  
  // capture point
  double* com_position = SensorByName(model, data, "pelvis_subtreecom");
  double* com_velocity = SensorByName(model, data, "pelvis_subtreelinvel");
  double capture_point[3] = {com_position[0], com_position[1], com_position[2]};

  //upper body com
  double com_position_upper[3] = {0, 0, 0};
  double mass_upper = mju_sum(model->body_mass+15, 91-15);  

  for (int i = 15; i < 91; i++) {
    mju_addToScl3(com_position_upper, data->xipos+3*i, model->body_mass[i]/mass_upper);
  }

  mju_copy(capture_point, com_position_upper, 3);

  mju_subFrom(fxy_avg, capture_point, 2);
  double com_feet_distance = mju_norm(fxy_avg, 2);
  residual[counter++] = com_feet_distance;

  // ----- Feet height ----- //
  residual[counter++] = 0.25 * (f1_position[2] + f2_position[2] + f3_position[2] + f4_position[2]);

  // ----- head vertical speed should near 0
  double* com_velocity_head = SensorByName(model, data, "head_subtreelinvel");
  residual[counter++] = com_velocity_head[2];

  // ----- penalize feet cross ----- //

  double* hip_l_position = SensorByName(model, data, "hip_l");
  double* hip_r_position = SensorByName(model, data, "hip_r");
  double hip_dir[2];
  mju_sub(hip_dir, hip_l_position, hip_r_position, 2);
  mju_normalize(hip_dir, 2);

  double left_feet_com[2];
  mju_add(left_feet_com, f1_position, f2_position, 2);
  mju_scl(left_feet_com, left_feet_com, 0.5, 2);
  double right_feet_com[2];
  mju_add(right_feet_com, f3_position, f4_position, 2);
  mju_scl(right_feet_com, right_feet_com, 0.5, 2);


  double feet_dir[2];
  mju_sub(feet_dir, f1_position, f3_position, 2); //calcn position
  // mju_normalize(feet_dir, 2);
  double toe_dir[2];
  mju_sub(toe_dir, f2_position, f4_position, 2);

  double* knee_l_position = SensorByName(model, data, "knee_l");
  double* knee_r_position = SensorByName(model, data, "knee_r");
  double knee_dir[2];
  mju_sub(knee_dir, knee_l_position, knee_r_position, 2);


  double feet_cross = mju_min(0, mju_dot(hip_dir, feet_dir, 2)-0.2);
  feet_cross += mju_min(0, mju_dot(hip_dir, toe_dir, 2)-0.2);
  feet_cross += mju_min(0, mju_dot(hip_dir, knee_dir, 2)-0.15);
  residual[counter++] = feet_cross;

  // ----- forward velocity ----- //
  double forward_dir[3] = {1, 0, 0};

  residual[counter++] = mju_dot(com_velocity, forward_dir, 2) - parameters_[1];

  // ----- forward velocity angle ----- //
  double cos_theta = mju_dot(com_velocity, forward_dir, 2) / (mju_norm(com_velocity, 2)+1e-7);
  double sin_theta = mju_sqrt(1 - cos_theta * cos_theta);
  double side_velocity = mju_norm(com_velocity, 2) * sin_theta;
  residual[counter++] = side_velocity;
  
  // ----- hip direction ----- //
  residual[counter++] = mju_dot(hip_dir, forward_dir, 2);

  // ----- penalize large feet distance ----- //
  double feet_vec[3];
  mju_sub(feet_vec, f1_position, f2_position, 3);
  double feet_len = mju_norm(feet_vec, 3);
  
  double feet_diff_vec[2];
  mju_sub(feet_diff_vec, left_feet_com, right_feet_com, 2);
  
  double feet_distance = mju_norm(feet_diff_vec, 2);
  residual[counter++] = mju_max(0, feet_distance-feet_len);

  // ----- action ----- //
  mju_copy(&residual[counter], data->ctrl, model->nu);
  counter += model->nu;

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
        "and actual length of residual %d %d",
        counter);
  }
}

void Walk2::TransitionLocked(mjModel* model, mjData* data) {
  if (model->nv >= 170) {
    mju_zero(data->qvel+85, 85);
  }
}
}