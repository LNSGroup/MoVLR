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

#include "mjpc/tasks/musculoskeletal/musculoskeletal.h"

#include <string>

#include <absl/random/random.h>
#include <mujoco/mujoco.h>
#include "mjpc/utilities.h"

namespace mjpc {
std::string Musculoskeletal::XmlPath() const {
  return GetModelPath("musculoskeletal/task.xml");
}
std::string Musculoskeletal::Name() const { return "Musculoskeletal"; }
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
void Musculoskeletal::ResidualFn::Residual(const mjModel* model, const mjData* data,
                       double* residual) const {

  int counter = 0;
  // ----- Height: head feet vertical error ----- //
  // feet sensor positions
  double* f1_position = SensorByName(model, data, "sp0");
  double* f2_position = SensorByName(model, data, "sp1");
  double* f3_position = SensorByName(model, data, "sp2");
  double* f4_position = SensorByName(model, data, "sp3");

  double* head_position = SensorByName(model, data, "head_position");
  double head_feet_error =
      head_position[2] - 0.25 * (f1_position[2] + f2_position[2] +
                                 f3_position[2] + f4_position[2]);
  residual[counter++] = head_feet_error - parameters_[0];

  // double* pelvis_position = SensorByName(model, data, "pelvis_position");
  // double pelvis_feet_error =
  //     pelvis_position[2] - 0.25 * (f1_position[2] + f2_position[2] +
  //                                f3_position[2] + f4_position[2]);
  // residual[counter++] = pelvis_feet_error - parameters_[0];
  // printf("head feet error %f\n", head_feet_error);
  // ----- Balance: CoM-feet xy error ----- //

  // //compute com
  // double total_mass = 0;
  // double com_position[3] = {0, 0, 0};
  // double com_velocity[3] = {0, 0, 0};

  
  // for (int i = 0; i < model->nbody; i++) {
  //   double mass = model->body_mass[i];
  //   total_mass += mass;
  //   double mass_pos[3] = {data->xipos[3*i], data->xipos[3*i+1], data->xipos[3*i+2]};
  //   double mass_vel[3] = {data->cvel[6*i+3], data->cvel[6*i+4], data->cvel[6*i+5]};

  //   mju_scl(mass_pos, mass_pos, mass, 3);
  //   mju_scl(mass_vel, mass_vel, mass, 3);
  //   mju_addTo(com_position, mass_pos, 3);
  //   mju_addTo(com_velocity, mass_vel, 3);

  // }


  
  // mju_scl(com_position, com_position, 1.0/total_mass, 3);
  // mju_scl(com_velocity, com_velocity, 1.0/total_mass, 3);



  
  // capture point
  // double* com_position = SensorByName(model, data, "head_subtreecom");
  // double* com_velocity = SensorByName(model, data, "head_subtreelinvel");
  double* com_position = SensorByName(model, data, "pelvis_subtreecom");
  double* com_velocity = SensorByName(model, data, "pelvis_subtreelinvel");
  double kFallTime = 0.2;
  double capture_point[3] = {com_position[0], com_position[1], com_position[2]};
  mju_addToScl3(capture_point, com_velocity, kFallTime);

  // average feet xy position
  double fxy_avg[2] = {0.0};
  mju_addTo(fxy_avg, f1_position, 2);
  mju_addTo(fxy_avg, f2_position, 2);
  mju_addTo(fxy_avg, f3_position, 2);
  mju_addTo(fxy_avg, f4_position, 2);
  mju_scl(fxy_avg, fxy_avg, 0.25, 2);

  // double* torso_position = SensorByName(model, data, "torso_position");

  mju_subFrom(fxy_avg, capture_point, 2);
  double com_feet_distance = mju_norm(fxy_avg, 2);
  residual[counter++] = com_feet_distance;
  // printf("com feet distance %f %f\n", fxy_avg[0], fxy_avg[1]);           
  
  // TODO: too slow for this computation
  // ----- head-feet horizontal distance be 0 ----- //
  // double* com_position_head = SensorByName(model, data, "head_subtreecom");
  // double* com_velocity_head = SensorByName(model, data, "head_subtreelinvel");
  // double kFallTime_head = 0.2;
  // double capture_point_head[3] = {com_position_head[0], com_position_head[1], com_position_head[2]};
  // mju_addToScl3(capture_point_head, com_velocity_head, kFallTime_head);

  // // average feet xy position
  // double fxy_avg_head[2] = {0.0};
  // mju_addTo(fxy_avg_head, f1_position, 2);
  // mju_addTo(fxy_avg_head, f2_position, 2);
  // mju_addTo(fxy_avg_head, f3_position, 2);
  // mju_addTo(fxy_avg_head, f4_position, 2);
  // mju_scl(fxy_avg_head, fxy_avg_head, 0.25, 2);

  // // double* torso_position = SensorByName(model, data, "torso_position");

  // mju_subFrom(fxy_avg_head, capture_point_head, 2);
  // double com_feet_distance_head = mju_norm(fxy_avg_head, 2);
  // residual[counter++] = com_feet_distance_head;
  residual[counter++] = 0;


  // // ----- head-pelvis horizontal distance be 0 ----- //
  // double pelvis_xy[2] = {com_position[0], com_position[1]};
  // mju_subFrom(pelvis_xy, capture_point_head, 2);
  // double com_distance_head_pelvis = mju_norm(pelvis_xy, 2);
  // residual[counter++] = com_distance_head_pelvis;
  residual[counter++] = 0;


  // ----- COM xy velocity should be 0 ----- //
  mju_copy(&residual[counter], com_velocity, 2);
  // mju_copy(&residual[counter], 0, 2);

  counter += 2;

  // ----- joint velocity ----- //
  mju_copy(residual + counter, data->qvel, model->nv);
  counter += model->nv;

  // ----- action ----- //
  mju_copy(&residual[counter], data->ctrl, model->nu);
  counter += model->nu;
  // printf("joint number %d\n", model->nv);

  
  // ----- disorder----- //
  //penalize muscle activate where the muscle length increase
  // double muscle_length = data->actuator_length;
  // double backward_activation = 0;
  // for (int i = 0; i < model->nsensor; i++) {
  //   double length_diff = data->actuator_length[i] - model->actuator_length0[i];
  //   if (length_diff > 0 && data->ctrl[i] > 0) {
  //     backward_activation = backward_activation + data->ctrl[i];
  //     // printf("backward activation: %f %d\n", data->ctrl[i], i);
  //   }
  // }
  // // printf("backward activation: %f\n", backward_activation);
  // residual[counter++] = backward_activation;
  // counter += 1;


  // printf("left foot pos %f %f %f %f\n", f1_position[0], f1_position[1], f2_position[0], f2_position[1]);
  // printf("right foot pos %f %f %f %f\n", f3_position[0], f3_position[1], f4_position[0], f4_position[1]);
  // printf("torso pos %f %f %f\n", torso_position[0], torso_position[1], torso_position[2]);
  
  // double min_x = f1_position[0] < f3_position[0]? f1_position[0]: f3_position[0];
  // double max_x = f2_position[0] >= f4_position[0]? f2_position[0]: f4_position[0];
  // double min_y = f1_position[1] < f2_position[1]? f1_position[1]: f2_position[1];
  // double max_y = f3_position[1] >= f4_position[1]? f3_position[1]: f4_position[1];
  
  // double min_x = f1_position[0];
  // double max_x = f1_position[0];

  // min_x = min_x < f2_position[0]? min_x: f2_position[0];
  // min_x = min_x < f3_position[0]? min_x: f3_position[0];
  // min_x = min_x < f4_position[0]? min_x: f4_position[0];

  // max_x = max_x >= f2_position[0]? max_x: f2_position[0];
  // max_x = max_x >= f3_position[0]? max_x: f3_position[0];
  // max_x = max_x >= f4_position[0]? max_x: f4_position[0];

  // double min_y = f1_position[1];
  // double max_y = f1_position[1];

  // min_y = min_y < f2_position[1]? min_y: f2_position[1];
  // min_y = min_y < f3_position[1]? min_y: f3_position[1];
  // min_y = min_y < f4_position[1]? min_y: f4_position[1];

  // max_y = max_y >= f2_position[1]? max_y: f2_position[1];
  // max_y = max_y >= f3_position[1]? max_y: f3_position[1];
  // max_y = max_y >= f4_position[1]? max_y: f4_position[1];
  
  
  // printf("minmaxpos %f %f %f %f\n", min_x, max_x, min_y, max_y);
  
  // printf("key number %d", modedouble* f1_position = SensorByName(model, data, "sp0");
  // mju_error_i(
  //       "mismatch between total user-sensor dimension "
  //       "and actual length of residual %d",
  //       counter);
  int user_sensor_dim = 0;
  for (int i = 0; i < model->nsensor; i++) {
    if (model->sensor_type[i] == mjSENS_USER) {
      user_sensor_dim += model->sensor_dim[i];
    }
  }
  // printf("user_sensor_dim: %d %d\n", user_sensor_dim, counter);
  if (user_sensor_dim != counter) {
    mju_error_i(
        "mismatch between total user-sensor dimension "
        "and actual length of residual %d",
        counter);
  }

 
  // printf("muscle length: %f\n", muscle_length);


}

// -------- Transition for swimmer task --------
//   If swimmer is within tolerance of goal ->
//   move goal randomly.
// ---------------------------------------------
void Musculoskeletal::TransitionLocked(mjModel* model, mjData* data) {
  // double* target = SensorByName(model, data, "target");
  // double* nose = SensorByName(model, data, "nose");
  // double nose_to_target[2];
  // mju_sub(nose_to_target, target, nose, 2);
  // if (mju_norm(nose_to_target, 2) < 0.04) {
  //   absl::BitGen gen_;
  //   data->mocap_pos[0] = absl::Uniform<double>(gen_, -.8, .8);
  //   data->mocap_pos[1] = absl::Uniform<double>(gen_, -.8, .8);
  // }
  // double* f1_position = SensorByName(model, data, "sp0");
  // double* f2_position = SensorByName(model, data, "sp1");
  // double* f3_position = SensorByName(model, data, "sp2");
  // double* f4_position = SensorByName(model, data, "sp3");
  // double* torso_position = SensorByName(model, data, "torso_position");
  
  //   double min_x = f1_position[0];
  // double max_x = f1_position[0];

  // min_x = min_x < f2_position[0]? min_x: f2_position[0];
  // min_x = min_x < f3_position[0]? min_x: f3_position[0];
  // min_x = min_x < f4_position[0]? min_x: f4_position[0];

  // max_x = max_x >= f2_position[0]? max_x: f2_position[0];
  // max_x = max_x >= f3_position[0]? max_x: f3_position[0];
  // max_x = max_x >= f4_position[0]? max_x: f4_position[0];

  // double min_y = f1_position[1];
  // double max_y = f1_position[1];

  // min_y = min_y < f2_position[1]? min_y: f2_position[1];
  // min_y = min_y < f3_position[1]? min_y: f3_position[1];
  // min_y = min_y < f4_position[1]? min_y: f4_position[1];

  // max_y = max_y >= f2_position[1]? max_y: f2_position[1];
  // max_y = max_y >= f3_position[1]? max_y: f3_position[1];
  // max_y = max_y >= f4_position[1]? max_y: f4_position[1];
  // // printf("curent height %f\n ", head_feet_error);
  // if (torso_position[0]<min_x || torso_position[0]>max_x || torso_position[1]<min_y || torso_position[1]>max_y) {
  //   printf("reset x pos%f %f %f\n", torso_position[0], min_x, max_x);
  //   printf("reset y pos %f %f %f\n", torso_position[1], min_y, max_y);

  //   // mju_copy(data->qpos, model->key_qpos, model->nq);
  //   mju_copy(data->qpos, model->qpos0, model->nq);

  //   mju_zero(data->qvel, model->nv);
  //   // mju_error_i(
  //   //     "com deviate from feet", 1
  //   //     );
  // }
}

}  // namespace mjpc
