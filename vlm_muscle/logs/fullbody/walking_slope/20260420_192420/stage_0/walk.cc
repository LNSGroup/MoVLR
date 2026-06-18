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

  //upper body com
  double com_position_upper[3] = {0, 0, 0};
  double com_velocity_upper[3] = {0, 0, 0};
  double mass_upper = mju_sum(model->body_mass, model->nbody);  
  for (int i = 0; i < model->nbody; i++) {
    mju_addToScl3(com_position_upper, data->xipos+3*i, model->body_mass[i]/mass_upper);
    mju_addToScl3(com_velocity_upper, data->cvel+6*i+3, model->body_mass[i]/mass_upper);
  }
 
  mju_copy(capture_point, com_position_upper, 3);

  mju_subFrom(fxy_avg, capture_point, 2);
  double com_feet_distance = mju_norm(fxy_avg, 2);
  residual[counter++] = com_feet_distance;

  // ----- Joint Pos (keep untouched due to simulator) ----- //
  residual[counter++] = mju_norm(data->qpos+6, 85-6);
  
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