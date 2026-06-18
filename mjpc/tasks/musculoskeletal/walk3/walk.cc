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

#include "mjpc/tasks/musculoskeletal/walk3/walk.h"

#include <string>

#include <absl/random/random.h>
#include <mujoco/mujoco.h>
#include "mjpc/utilities.h"

namespace mjpc::musculoskeletal {

std::string Walk3::XmlPath() const {
  return GetModelPath("musculoskeletal/walk3/task.xml");
}
std::string Walk3::Name() const { return "MS Walk3"; }
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
void Walk3::ResidualFn::Residual(const mjModel* model, const mjData* data,
                       double* residual) const {

  int counter = 0;
  // ----- Height: head feet vertical error ----- //
  // feet sensor positions
  double* f1_position = SensorByName(model, data, "sp0");
  double* f2_position = SensorByName(model, data, "sp1");
  double* f3_position = SensorByName(model, data, "sp2");
  double* f4_position = SensorByName(model, data, "sp3");
  // std::cout<<"feet height "<<f1_position[1]<<" "<<f2_position[1]<<" "<<f3_position[1]<<" "<<f4_position[1]<<std::endl;
  double* head_position = SensorByName(model, data, "head_position");
  double feet_height = 0;
  double fxy_avg[2] = {0, 0};
  // double fz = 0;
  // double onground_cnt = 0;
  // if (f1_position[2] <= 0.02) {
  //   feet_height += f1_position[2];
  //   fxy_avg[0] += f1_position[0];
  //   fxy_avg[1] += f1_position[1];
  //   onground_cnt += 1;
  // }
  // if (f2_position[2] <= 0.02) {
  //   feet_height += f2_position[2];
  //   fxy_avg[0] += f2_position[0];
  //   fxy_avg[1] += f2_position[1];
  //   onground_cnt += 1;
  // }
  // if (f3_position[2] <= 0.02) {
  //   feet_height += f3_position[2];
  //   fxy_avg[0] += f3_position[0];
  //   fxy_avg[1] += f3_position[1];
  //   onground_cnt += 1;
  // }
  // if (f4_position[2] <= 0.02) {
  //   feet_height += f4_position[2];
  //   fxy_avg[0] += f4_position[0];
  //   fxy_avg[1] += f4_position[1];
  //   onground_cnt += 1;
  // }

  // if (onground_cnt == 0) {
  //   feet_height = 0.25 * (f1_position[2] + f2_position[2] + f3_position[2] + f4_position[2]);
  //   fxy_avg[0] = 0.25 * (f1_position[0] + f2_position[0] + f3_position[0] + f4_position[0]);
  //   fxy_avg[1] = 0.25 * (f1_position[1] + f2_position[1] + f3_position[1] + f4_position[1]);
  // } 
  // else {
  //   feet_height /= onground_cnt;
  //   fxy_avg[0] /= onground_cnt;
  //   fxy_avg[1] /= onground_cnt;
  // }

  


  // if (f1_position[2] > 0.02 and f2_position[2] > 0.02) {
  //   feet_height = 0.5 * (f3_position[2] + f4_position[2]);
  //   fxy_avg[0] = 0.5 * (f3_position[0] + f4_position[0]);
  //   fxy_avg[1] = 0.5 * (f3_position[1] + f4_position[1]);
  // } else if (f3_position[2] > 0.02 and f4_position[2] > 0.02) {
  //   feet_height = 0.5 * (f1_position[2] + f2_position[2]);
  //   fxy_avg[0] = 0.5 * (f1_position[0] + f2_position[0]);
  //   fxy_avg[1] = 0.5 * (f1_position[1] + f2_position[1]);
  // } else {
  feet_height = 0.25 * (f1_position[2] + f2_position[2] + f3_position[2] + f4_position[2]);
  fxy_avg[0] = 0.25 * (f1_position[0] + f2_position[0] + f3_position[0] + f4_position[0]);
  fxy_avg[1] = 0.25 * (f1_position[1] + f2_position[1] + f3_position[1] + f4_position[1]);
  // }



  // double head_feet_error =
  //     head_position[2] - 0.25 * (f1_position[2] + f2_position[2] +
  //                                f3_position[2] + f4_position[2]);
  double head_feet_error =
      head_position[2] - feet_height;
  residual[counter++] = head_feet_error - parameters_[0];

  // std::cout<<head_feet_error<<std::endl;

  
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

  // ----- upright ----- //
  double up_dir[3] = {0*(data->qpos[0]-parameters_[0]), 0, 1};

  double* torso_up = SensorByName(model, data, "torso_up");
  double* pelvis_up = SensorByName(model, data, "pelvis_up");
  double* foot_left_up = SensorByName(model, data, "foot_left_up");
  double* foot_right_up = SensorByName(model, data, "foot_right_up");

  double upright = 0;

  upright += 1 * (1- mju_dot(torso_up, up_dir, 3));
  upright += 1 * (1- mju_dot(pelvis_up, up_dir, 3));
  upright += 0.1 * (1- mju_dot(foot_left_up, up_dir, 3));
  upright += 0.1 * (1- mju_dot(foot_right_up, up_dir, 3));
  residual[counter++] = upright;
  
  // capture point
  // double* com_position = SensorByName(model, data, "head_subtreecom");
  // double* com_velocity = SensorByName(model, data, "head_subtreelinvel");
  double* com_position = SensorByName(model, data, "pelvis_subtreecom");
  double* com_velocity = SensorByName(model, data, "pelvis_subtreelinvel");
  // double kFallTime = 0.0;
  double capture_point[3] = {com_position[0], com_position[1], com_position[2]};
  // mju_addToScl3(capture_point, com_velocity, kFallTime);
  // mju_addTo3(capture_point, head_position);
  // mju_scl3(capture_point, capture_point, 0.5);


  //upper body com
  double com_position_upper[3] = {0, 0, 0};
  double com_velocity_upper[3] = {0, 0, 0};
  double mass_upper = mju_sum(model->body_mass+15, 91-15);  
  // std::cout<<"mass upper "<<mass_upper<<std::endl;
  for (int i = 15; i < 91; i++) {
    mju_addToScl3(com_position_upper, data->xipos+3*i, model->body_mass[i]/mass_upper);
    mju_addToScl3(com_velocity_upper, data->cvel+6*i+3, model->body_mass[i]/mass_upper);
  }
  // std::cout<<"com upper "<<com_position_upper[0]<<" "<<com_position_upper[1]<<" "<<com_position_upper[2]<<std::endl;
  // mju_addTo3(capture_point, com_position_upper);
  // mju_scl3(capture_point, capture_point, 0.5);
  mju_copy(capture_point, com_position_upper, 3);
  double kFallTime = 0.01;


  // mju_scl3(com_velocity_upper, {parameters_[1]}, 1);
  com_position_upper[0] -= parameters_[1];
  mju_addToScl3(capture_point, com_velocity_upper, kFallTime);
  // std::cout<<"com upper "<<com_position_upper[0]<<" "<<com_position_upper[1]<<" "<<com_position_upper[2]<<std::endl;

  // average feet xy position
  // double fxy_avg[2] = {0.0};
  // mju_addTo(fxy_avg, f1_position, 2);
  // mju_addTo(fxy_avg, f2_position, 2);
  // mju_addTo(fxy_avg, f3_position, 2);
  // mju_addTo(fxy_avg, f4_position, 2);
  // mju_scl(fxy_avg, fxy_avg, 0.25, 2);

  // double* torso_position = SensorByName(model, data, "torso_position");
  // fxy_avg[0] += 0.2*parameters_[1];
  mju_subFrom(fxy_avg, capture_point, 2);
  double com_feet_distance = mju_norm(fxy_avg, 2);
  residual[counter++] = com_feet_distance;

  // ----- Feet height ----- //
  // residual[counter++] = 0.25 * (f1_position[2] + f2_position[2] + f3_position[2] + f4_position[2]);
  // printf("com feet distance %f %f\n", fxy_avg[0], fxy_avg[1]);           
  
  
  // // ----- head-feet com should match ----- //
  // double* com_position_head = SensorByName(model, data, "head_position");
  // double com_head_feet[2];
  // mju_sub(com_head_feet, com_position_head, fxy_avg, 2);
  // residual[counter++] = mju_norm(com_head_feet, 2);

  // ----- head vertical speed should near 0
  // double* com_velocity_head = SensorByName(model, data, "head_subtreelinvel");
  // residual[counter++] = com_velocity_head[2];


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

  
  // residual[counter++] = 0;
  // ----- penalize feet cross ----- //

  double* hip_l_position = SensorByName(model, data, "hip_l");
  double* hip_r_position = SensorByName(model, data, "hip_r");
  double hip_dir[2];
  mju_sub(hip_dir, hip_l_position, hip_r_position, 2);
  // std::cout<<mju_norm(hip_dir, 2)<<std::endl;
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
  // std::cout<<"dot product "<<mju_dot(hip_dir, feet_dir, 2)<<" "<<mju_dot(hip_dir, toe_dir, 2)<<" "<<mju_dot(hip_dir, knee_dir, 2)<<" "<<feet_cross<<std::endl;
  
  // double feet_dir_l[2];
  // mju_sub(feet_dir_l, left_feet_com, hip_l_position, 2);
  // double feet_dir_r[2];
  // mju_sub(feet_dir_r, right_feet_com, hip_r_position, 2);
  // double feet_cross = mju_min(0, mju_dot(hip_dir, feet_dir_l, 2));
  // feet_cross += mju_min(0, -mju_dot(hip_dir, feet_dir_r, 2));
  // std::cout<<"feet cross "<<mju_dot(hip_dir, feet_dir_l, 2)<<" "<<-mju_dot(hip_dir, feet_dir_r, 2)<<std::endl;
  residual[counter++] = feet_cross;

  // std::cout<<"feet cross "<<feet_cross<<std::endl;


  // // ----- head-pelvis horizontal distance be 0 ----- //
  // double pelvis_xy[2] = {com_position[0], com_position[1]};
  // mju_subFrom(pelvis_xy, capture_point_head, 2);
  // double com_distance_head_pelvis = mju_norm(pelvis_xy, 2);
  // residual[counter++] = com_distance_head_pelvis;
  // residual[counter++] = 0;


  // ----- COM xy velocity should be 0 ----- //
  // mju_copy(&residual[counter], com_velocity, 2);
  // // mju_copy(&residual[counter], 0, 2);

  // counter += 2;
  // ----- forward velocity ----- //
  double forward_dir[3] = {1, 0, 0};

  residual[counter++] = mju_dot(com_velocity, forward_dir, 2) - parameters_[1];

  // ----- forward velocity angle ----- //
  double cos_theta = mju_dot(com_velocity, forward_dir, 2) / (mju_norm(com_velocity, 2)+1e-7);
  double sin_theta = mju_sqrt(1 - cos_theta * cos_theta);
  double side_velocity = mju_norm(com_velocity, 2) * sin_theta;
  residual[counter++] = side_velocity;
  // std::cout<<"side velocity "<<side_velocity<<" "<<sin_theta<<" "<<mju_norm(com_velocity, 2)<<std::endl;
  
  // ----- forward ----- //
  // residual[counter++] = mju_dot(hip_dir, forward_dir, 2);

  double* torso_forward = SensorByName(model, data, "torso_forward");
  double* pelvis_forward = SensorByName(model, data, "pelvis_forward");
  double* knee_left_forward = SensorByName(model, data, "knee_left_forward");
  double* knee_right_forward = SensorByName(model, data, "knee_right_forward");

  double forward = 0;

  forward += 0 * (1- mju_dot(torso_forward, forward_dir, 3));
  forward += 1 * (1- mju_dot(pelvis_forward, forward_dir, 3));
  forward += 0. * (1- mju_dot(knee_left_forward, forward_dir, 3));
  forward += 0. * (1- mju_dot(knee_right_forward, forward_dir, 3));
  residual[counter++] = forward;

  // ----- penalize large feet distance ----- //
  // double feet_vec[3];
  // mju_sub(feet_vec, f1_position, f2_position, 3);
  // double feet_len = mju_norm(feet_vec, 3);
  
  // double feet_diff_vec[2];
  // mju_sub(feet_diff_vec, left_feet_com, right_feet_com, 2);
  
  // double feet_distance = mju_norm(feet_diff_vec, 2);
  // // cos_theta = mju_dot(feet_diff_vec, hip_dir, 2) / (mju_norm(feet_diff_vec, 2)+1e-7);
  // // sin_theta = mju_sqrt(1 - cos_theta * cos_theta);
  // // double feet_distance = mju_norm(feet_diff_vec, 2) * sin_theta;
  // residual[counter++] = mju_max(0, feet_distance-feet_len);
  // residual[counter++] = feet_distance;

  // ----- joint velocity ----- //
  // mju_copy(residual + counter, data->qvel, 85);
  // counter += 85;

  // ----- action ----- //
  // mju_copy(&residual[counter], data->ctrl, model->nu);
  // counter += model->nu;
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
    std::cout<<"sensor dim"<<user_sensor_dim<<" "<<counter<<std::endl;
    mju_error_i(
        "mismatch between total user-sensor dimension "
        "and actual length of residual %d %d",
        counter);
  }

 

}

void Walk3::TransitionLocked(mjModel* model, mjData* data) {
  // double* target = SensorByName(model, data, "target");
  // double* nose = SensorByName(model, data, "nose");
  // double nose_to_target[2];
  // double torso_height = SensorByName(model, data, "torso_position")[2];
  // mju_sub(nose_to_target, target, nose, 2);
  // if (data->time == 0) {
  //   mju_copy(data->qpos, model->key_qpos, model->nq);
  //   // std::cout<<"reset "<<data->qpos[3]<<std::endl;
  //   mju_zero(data->qvel, model->nv);
  //   // mju_zero(data->act, model->na);
  // }
  // mju_copy(data->qpos+85, model->key_qpos, 85);
  // std::cout<<"key "<<model->key_qpos[6]<<std::endl;
  // if (data->time == 0) {
  //   mju_copy(data->qpos, model->key_qpos, model->nq);
  //   // std::cout<<"reset "<<data->qpos[3]<<std::endl;
  //   mju_zero(data->qvel, model->nv);
  //   // mju_zero(data->act, model->na);
  // }
  if (model->nv >= 170) {
    mju_zero(data->qvel+85, 85);
  }
  
  // std::cout<<model->nuserdata<<std::endl;
}

}  // namespace mjpc
