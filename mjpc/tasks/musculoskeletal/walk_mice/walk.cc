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

#include "mjpc/tasks/musculoskeletal/walk_mice/walk.h"

#include <string>

#include <absl/random/random.h>
#include <mujoco/mujoco.h>
#include "mjpc/utilities.h"

namespace mjpc::musculoskeletal {

std::string WalkMice::XmlPath() const {
  return GetModelPath("musculoskeletal/walk_mice/task.xml");
}
std::string WalkMice::Name() const { return "MS WalkMice"; }
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
void WalkMice::ResidualFn::Residual(const mjModel* model, const mjData* data,
                       double* residual) const {

  
  
  int counter = 0;
  // residual[counter++] = 0;
  // ----- Height: head feet vertical error ----- //
  // feet sensor positions
  double* f1_position = SensorByName(model, data, "sp0");
  double* f11_position = SensorByName(model, data, "sp00");
  double* f2_position = SensorByName(model, data, "sp1");
  double* f3_position = SensorByName(model, data, "sp2");
  double* f33_position = SensorByName(model, data, "sp22");
  double* f4_position = SensorByName(model, data, "sp3");
  // std::cout<<"feet height "<<f1_position[1]<<" "<<f2_position[1]<<" "<<f3_position[1]<<" "<<f4_position[1]<<std::endl;
  double* head_position = SensorByName(model, data, "head_position");
  double* tail_position = SensorByName(model, data, "tail_position");
  // double feet_height = 0;
  double fxy_avg[2] = {0, 0};
  
  double feet_height1 = 0.25 * (f1_position[2] + f3_position[2] + f11_position[2] + f33_position[2]);
  double feet_height2 = 0.5 * (f2_position[2] + f4_position[2]);
  // feet_height = mju_min(mju_min(f1_position[2] ,f2_position[2]),  mju_min(f3_position[2] ,f4_position[2]));
  fxy_avg[0] = 0.25 * (f1_position[0] + f2_position[0] + f3_position[0] + f4_position[0]);
  fxy_avg[1] = 0.25 * (f1_position[1] + f2_position[1] + f3_position[1] + f4_position[1]);
  // }
  double head_feet_error =
      // (mju_min(parameters_[0], (head_position[2] - feet_height)) + mju_min(parameters_[0], (tail_position[2] - feet_height)))/2;
      (head_position[2] - feet_height2- parameters_[0] + tail_position[2] - feet_height1- parameters_[0] + 0.005)/2;
  residual[counter++] = head_feet_error;

  // std::cout<<"head feet error "<<head_feet_error<<std::endl;

  //----- head > tail ------//
  residual[counter++] = mju_min(0, head_position[2] - tail_position[2]);

  // ----- upright ----- //
  double up_dir[3] = {0*(data->qpos[0]-parameters_[0]), 0, 1};

  double* torso_up = SensorByName(model, data, "torso_up");
  double* pelvis_up = SensorByName(model, data, "pelvis_up");
  // double* head_up = SensorByName(model, data, "head_up");
  // double* foot_left_up = SensorByName(model, data, "foot_left_up");
  // double* foot_right_up = SensorByName(model, data, "foot_right_up");
  double* toe_left_up = SensorByName(model, data, "toe_left_up");
  double* toe_right_up = SensorByName(model, data, "toe_right_up");

  double upright = 0;

  upright += 1 * (1- mju_dot(torso_up+1, up_dir+1, 2));
  upright += 1 * (1- mju_dot(pelvis_up+1, up_dir+1, 2));
  // upright += 1 * (1- mju_dot(head_up, up_dir, 3));
  // upright += 0.1 * (1- mju_dot(foot_left_up+1, up_dir+1, 2));
  // upright += 0.1 * (1- mju_dot(foot_right_up+1, up_dir+1, 2));
  upright += 0.1 * (1- mju_dot(toe_left_up+1, up_dir+1, 2));
  upright += 0.1 * (1+ mju_dot(toe_right_up+1, up_dir+1, 2));
  residual[counter++] = upright;
  
  
  double* com_position = SensorByName(model, data, "pelvis_subtreecom");
  double* com_velocity = SensorByName(model, data, "pelvis_subtreelinvel");
 
  double capture_point[3] = {com_position[0], com_position[1], com_position[2]};


  //upper body com
  double com_position_upper[3] = {0, 0, 0};
  double com_velocity_upper[3] = {0, 0, 0};
  double mass_upper = mju_sum(model->body_mass, model->nbody);  
  // std::cout<<"mass upper "<<mass_upper<<std::endl;
  for (int i = 0; i < model->nbody; i++) {
    mju_addToScl3(com_position_upper, data->xipos+3*i, model->body_mass[i]/mass_upper);
    mju_addToScl3(com_velocity_upper, data->cvel+6*i+3, model->body_mass[i]/mass_upper);
  }
 
  mju_copy(capture_point, com_position_upper, 3);
  // double kFallTime = 0.01;


  // mju_scl3(com_velocity_upper, {parameters_[1]}, 1);
  // com_position_upper[0] -= parameters_[1];
  // mju_addToScl3(capture_point, com_velocity_upper, kFallTime);
  
  mju_subFrom(fxy_avg, capture_point, 2);
  double com_feet_distance = mju_norm(fxy_avg, 2);
  residual[counter++] = com_feet_distance;

  
  // ----- Feet cross ----- //

  double* hip_l_position = SensorByName(model, data, "hip_l");
  double* hip_r_position = SensorByName(model, data, "hip_r");
  double hip_dir[2];
  mju_sub(hip_dir, hip_l_position, hip_r_position, 2);
  // std::cout<<mju_norm(hip_dir, 2)<<std::endl;
  mju_normalize(hip_dir, 2);

  // double left_feet_com[2];
  // mju_add(left_feet_com, f1_position, f2_position, 2);
  // mju_scl(left_feet_com, left_feet_com, 0.5, 2);
  // double right_feet_com[2];
  // mju_add(right_feet_com, f3_position, f4_position, 2);
  // mju_scl(right_feet_com, right_feet_com, 0.5, 2);


  double feet_dir[2];
  mju_sub(feet_dir, f1_position, f3_position, 2); //calcn position
  // mju_normalize(feet_dir, 2);
  double toe_dir[2];
  mju_sub(toe_dir, f2_position, f4_position, 2);
  // mju_normalize(toe_dir, 2);

  // double* knee_l_position = SensorByName(model, data, "knee_l");
  // double* knee_r_position = SensorByName(model, data, "knee_r");
  // double knee_dir[2];
  // mju_sub(knee_dir, knee_l_position, knee_r_position, 2);
  // std::cout<<mju_dot(hip_dir, feet_dir, 2)<<" "<<mju_dot(hip_dir, toe_dir, 2)<<std::endl;

  double feet_cross = mju_min(0, mju_dot(hip_dir, feet_dir, 2)-0.01);
  feet_cross += mju_min(0, mju_dot(hip_dir, toe_dir, 2)-0.01);
  // feet_cross += mju_min(0, mju_dot(hip_dir, knee_dir, 2)-0.15);
  
  residual[counter++] = feet_cross;

  
  // ----- Forward velocity ----- //
  double forward_dir[3] = {1, 0, 0};

  residual[counter++] = mju_min(0, mju_dot(com_velocity, forward_dir, 2) - parameters_[1]);
  // mju_min(0, mju_dot(com_velocity, forward_dir, 2) - parameters_[1]);

  // ----- Forward velocity angle ----- //
  double cos_theta = mju_dot(com_velocity, forward_dir, 3) / (mju_norm(com_velocity, 3)+1e-7);
  double sin_theta = mju_sqrt(1 - cos_theta * cos_theta);
  double side_velocity = mju_norm(com_velocity, 3) * sin_theta;
  residual[counter++] = side_velocity;
  // std::cout<<"side velocity "<<side_velocity<<" "<<sin_theta<<" "<<mju_norm(com_velocity, 2)<<std::endl;
  
  // ----- forward ----- //
  // residual[counter++] = mju_dot(hip_dir, forward_dir, 2);

  double* pelvis_forward = SensorByName(model, data, "pelvis_forward");
  double* foot_left_forward = SensorByName(model, data, "foot_left_forward");
  double* foot_right_forward = SensorByName(model, data, "foot_right_forward");
  double* toe_left_forward = SensorByName(model, data, "toe_left_forward");
  double* toe_right_forward = SensorByName(model, data, "toe_right_forward");



  double forward = 0;


  forward += 1 * (1- mju_dot(pelvis_forward, forward_dir, 3));
  forward += 1 * (1- mju_dot(foot_left_forward, forward_dir, 3));
  forward += 1 * (1- mju_dot(foot_right_forward, forward_dir, 3));
  forward += .1 * (1+ mju_dot(toe_left_forward, forward_dir, 3));
  forward += .1 * (1+ mju_dot(toe_right_forward, forward_dir, 3));


  residual[counter++] = forward;

  // // ----- Joint Pos ----- //
  // residual[counter++] = mju_norm(data->qpos+45, 85-45);

  
  
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

void WalkMice::TransitionLocked(mjModel* model, mjData* data) {
  // double* target = SensorByName(model, data, "target");
  // double* nose = SensorByName(model, data, "nose");
  // double nose_to_target[2];
  // double torso_height = SensorByName(model, data, "torso_position")[2];
  // mju_sub(nose_to_target, target, nose, 2);
  if (data->time == 0) {
    mju_copy(data->qpos, model->key_qpos, model->nq);
    // std::cout<<"reset "<<data->qpos[3]<<std::endl;
    mju_zero(data->qvel, model->nv);
    // mju_zero(data->act, model->na);
  }
  // mju_copy(data->qpos+85, model->key_qpos, 85);
  // std::cout<<"key "<<model->key_qpos[6]<<std::endl;
  // if (data->time == 0) {
  //   mju_copy(data->qpos, model->key_qpos+450*model->nq, model->nq);
  //   mju_copy(data->qvel, model->key_qvel+450*model->nv, model->nv);
  // }
  // double* f1_position = SensorByName(model, data, "sp0");
  // double* f2_position = SensorByName(model, data, "sp1");
  // double* f3_position = SensorByName(model, data, "sp2");
  // double* f4_position = SensorByName(model, data, "sp3");
  // double* head_position = SensorByName(model, data, "head_position");

  // // int current_index = static_cast<int>(ceil(data->time/0.002));
  // // mju_copy(data->qpos, model->key_qpos+current_index*model->nq, model->nq);
  // // mju_copy(data->qvel, model->key_qvel+current_index*model->nv, model->nv);
  // double feet_height = 0.25 * (f1_position[2] + f2_position[2] + f3_position[2] + f4_position[2]);
 
  // // // // }
  // double head_feet_error =
  //     head_position[2] - feet_height;
  
  // if (head_feet_error < 0.8)
  // {
  //   mju_copy(data->qpos, model->qpos0, model->nq);
  //   mju_zero(data->qvel, model->nv);
  // } 
  // std::cout<<"head feet error "<<head_feet_error<<std::endl;
  // if (model->nv >= 170) {
  //   mju_zero(data->qvel+85, 85);
  // }
  
  // std::cout<<model->nuserdata<<std::endl;
}

}  // namespace mjpc
