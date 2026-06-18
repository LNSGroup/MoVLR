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

#include "mjpc/tasks/musculoskeletal/swim/walk.h"

#include <string>

#include <absl/random/random.h>
#include <mujoco/mujoco.h>
#include "mjpc/utilities.h"

namespace mjpc::musculoskeletal {

std::string Swim::XmlPath() const {
  return GetModelPath("musculoskeletal/swim/task.xml");
}
std::string Swim::Name() const { return "MS Swim"; }
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
void Swim::ResidualFn::Residual(const mjModel* model, const mjData* data,
                       double* residual) const {

  int counter = 0;
  // ----- Height: head feet vertical error ----- //
  // feet sensor positions
  // double* f1_position = SensorByName(model, data, "sp0");
  // double* f2_position = SensorByName(model, data, "sp1");
  // double* f3_position = SensorByName(model, data, "sp2");
  // double* f4_position = SensorByName(model, data, "sp3");
  // // std::cout<<"feet height "<<f1_position[1]<<" "<<f2_position[1]<<" "<<f3_position[1]<<" "<<f4_position[1]<<std::endl;
  double* head_position = SensorByName(model, data, "head_position");
  // double feet_height = 0;
  // double fxy_avg[2] = {0, 0};
  
  // feet_height = 0.25 * (f1_position[2] + f2_position[2] + f3_position[2] + f4_position[2]);
  // fxy_avg[0] = 0.25 * (f1_position[0] + f2_position[0] + f3_position[0] + f4_position[0]);
  // fxy_avg[1] = 0.25 * (f1_position[1] + f2_position[1] + f3_position[1] + f4_position[1]);
  // // }
  // double head_feet_error =
  //     head_position[2];
      //  - feet_height;
  // residual[counter++] = head_feet_error - parameters_[0];
  // residual[counter++] = feet_height - parameters_[0];
  double pelvis_height = SensorByName(model, data, "pelvis_position")[2];
  residual[counter++] = pelvis_height - parameters_[0] + head_position[2] - parameters_[0];

  double* pelvis_vel = SensorByName(model, data, "pelvis_subtreelinvel");
  residual[counter++] = pelvis_vel[0] - parameters_[1];

  residual[counter++] = data->qpos[5];

  
  int user_sensor_dim = 0;
  for (int i = 0; i < model->nsensor; i++) {
    if (model->sensor_type[i] == mjSENS_USER) {
      user_sensor_dim += model->sensor_dim[i];
    }
  }
  // printf("user_sensor_dim: %d %d\n", user_sensor_dim, counter);
  if (user_sensor_dim != counter) {
    std::cout<<"sensor dim"<<user_sensor_dim<<" "<<counter<<std::endl;
    mju_error_i(
        "mismatch between total user-sensor dimension "
        "and actual length of residual %d %d",
        counter);
  }

 

}

void Swim::TransitionLocked(mjModel* model, mjData* data) {
  // double* target = SensorByName(model, data, "target");
  // double* nose = SensorByName(model, data, "nose");
  // double nose_to_target[2];
  // double torso_height = SensorByName(model, data, "torso_position")[2];
  // mju_sub(nose_to_target, target, nose, 2);
  // if (data->time == 0) {
  //   mju_copy(data->qpos, model->key_qpos, model->nq);
  //   // std::cout<<"reset "<<data->qpos[3]<<std::endl;
  //   mju_zero(data->qvel, model->nv);

  //   data->qpos[3] += 0.2;
  //   // mju_zero(data->act, model->na);
  // }
  // mju_copy(data->qpos+85, model->key_qpos, 85);
  // std::cout<<"key "<<model->key_qpos[6]<<std::endl;
  if (data->time == 0) {
    // data->qpos[1] = 0;
    data->qpos[3] = -1.57;
    mju_zero(data->qvel, model->nv);
    
    
  }
  if (model->nv >= 170) {
    mju_zero(data->qvel+85, 85);
  }
  // std::cout<<model->nq<<std::endl;
  // std::cout<<model->nuserdata<<std::endl;
}

}  // namespace mjpc
