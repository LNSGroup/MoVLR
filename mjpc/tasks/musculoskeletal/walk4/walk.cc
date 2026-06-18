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

#include "mjpc/tasks/musculoskeletal/walk4/walk.h"

#include <string>

#include <absl/random/random.h>
#include <mujoco/mujoco.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include "mjpc/utilities.h"

namespace mjpc::musculoskeletal {

struct Point {
    double x, y;
};

bool comparePoints(Point p1, Point p2) {
    return std::atan2(p1.y, p1.x) < std::atan2(p2.y, p2.x);
}

void sortPoints(std::vector<Point>& points) {
    std::sort(points.begin(), points.end(), comparePoints);
}

double area(Point p1, Point p2, Point p3) {
    return std::abs((p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x)) / 2.0;
}

double distanceFromPointToLine(Point p1, Point p2, Point p3) {
    double x1 = p1.x;
    double y1 = p1.y;
    double x2 = p2.x;
    double y2 = p2.y;
    double x3 = p3.x;
    double y3 = p3.y;
    double numerator = std::abs((y2 - y1) * x3 - (x2 - x1) * y3 + x2 * y1 - y2 * x1);
    double denominator = std::sqrt(std::pow(y2 - y1, 2) + std::pow(x2 - x1, 2));
    return numerator / denominator;
}

std::string Walk4::XmlPath() const {
  return GetModelPath("musculoskeletal/walk4/task.xml");
}
std::string Walk4::Name() const { return "MS Walk4"; }
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
void Walk4::ResidualFn::Residual(const mjModel* model, const mjData* data,
                       double* residual) const {
  // std::cout<<"con "<<data->ncon<<std::endl;
  // std::cout<<"con "<<data->efc_force[0]<<" "<<data->nefc<<std::endl;
  
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
  // feet_height = 1e9;
  // feet_height = mju_min(feet_height, f1_position[2]);
  // feet_height = mju_min(feet_height, f2_position[2]);
  // feet_height = mju_min(feet_height, f3_position[2]);
  // feet_height = mju_min(feet_height, f4_position[2]);
  feet_height = 0.25 * (f1_position[2] + f2_position[2] + f3_position[2] + f4_position[2]);
  fxy_avg[0] = 0.25 * (f1_position[0] + f2_position[0] + f3_position[0] + f4_position[0]);
  fxy_avg[1] = 0.25 * (f1_position[1] + f2_position[1] + f3_position[1] + f4_position[1]);
  // }



  // double head_feet_error =
  //     head_position[2] - 0.25 * (f1_position[2] + f2_position[2] +
  //                                f3_position[2] + f4_position[2]);
  double head_feet_error =
      head_position[2]  - feet_height;
  residual[counter++] = mju_min(head_feet_error - parameters_[0], 0);

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


  //upper body com
  // double com_position_upper[3] = {0, 0, 0};
  // // double com_velocity_upper[3] = {0, 0, 0};
  // double mass_upper = mju_sum(model->body_mass+15, model->nbody-15);  
  // // // std::cout<<"mass upper "<<mass_upper<<std::endl;
  // for (int i = 15; i < model->nbody; i++) {
  //   mju_addToScl3(com_position_upper, data->xipos+3*i, model->body_mass[i]/mass_upper);
  //   // mju_addToScl3(com_velocity_upper, data->cvel+6*i+3, model->body_mass[i]/mass_upper);
  //   // std::cout<<i<<" "<<mj_id2name(model, mjOBJ_BODY, i)<<std::endl;
  // }
  double* com_position = SensorByName(model, data, "pelvis_subtreecom");

  double kFallTime = 0.2;
  // ----- upright ----- //
  double* com_velocity = SensorByName(model, data, "pelvis_subtreelinvel");
  // std::cout<<"com vel "<<com_velocity[0]<<" "<<com_velocity[1]<<" "<<com_velocity[2]<<std::endl;
  double up_dir[3] = {kFallTime*(parameters_[1] - com_velocity[0]), 0, com_position[2]};
  mju_normalize3(up_dir);
  double* torso_up = SensorByName(model, data, "torso_up");
  double* pelvis_up = SensorByName(model, data, "pelvis_up");
  double* foot_left_up = SensorByName(model, data, "foot_left_up");
  double* foot_right_up = SensorByName(model, data, "foot_right_up");
  double* knee_left_up = SensorByName(model, data, "knee_left_up");
  double* knee_right_up = SensorByName(model, data, "knee_right_up");
  // double* head_up = SensorByName(model, data, "head_up");

  double upright = 0;

  double knee_up = mju_max(mju_dot(knee_left_up, up_dir, 3), mju_dot(knee_right_up, up_dir, 3));
  double foot_up = mju_max(mju_dot(foot_left_up, up_dir, 3), mju_dot(foot_right_up, up_dir, 3));
  // mju_add(knee_up, knee_left_up, knee_right_up, 3);
  // mju_scl(knee_up, knee_up, 0.5, 3);

  // double foot_up[2];
  // mju_add(foot_up, foot_left_up, foot_right_up, 3);
  // mju_scl3(foot_up, foot_up, 0.5);

  upright += 1 * (1- mju_dot(torso_up, up_dir, 3));
  upright += 1 * (1- mju_dot(pelvis_up, up_dir, 3));
  upright += 0 * (1- knee_up);
  upright += 0.1 * (1- foot_up);
  // upright += 0 * (1- mju_dot(knee_up, up_dir, 3));
  // upright += 0 * (1- mju_dot(foot_up, up_dir, 3));
  upright += 0 * (1- mju_dot(foot_left_up, up_dir, 3));
  upright += 0 * (1- mju_dot(foot_right_up, up_dir, 3));
  // upright += 0 * (1- mju_dot(head_up, up_dir, 3));
  // upright += 1 * (1- mju_dot(torso_up, pelvis_up, 3));
  residual[counter++] = upright;
  
  // capture point
  // double* com_position = SensorByName(model, data, "head_subtreecom");
  // double* com_velocity = SensorByName(model, data, "head_subtreelinvel");
  
  
  // double kFallTime = 0.0;
  double capture_point[3] = {com_position[0], com_position[1], com_position[2]};
  // mju_addToScl3(capture_point, com_velocity, kFallTime);
  // mju_addTo3(capture_point, head_position);
  // mju_scl3(capture_point, capture_point, 0.5);


  std::vector<Point> hull = {
    {f1_position[0], f1_position[1]},
    {f2_position[0], f2_position[1]},
    {f3_position[0], f3_position[1]},
    {f4_position[0], f4_position[1]}
  };

  sortPoints(hull);
  

  
  Point p = {capture_point[0], capture_point[1]};
  double hullArea = 0.0;
  for (int i = 0; i < hull.size(); i++) {
      hullArea += area(hull[i], hull[(i+1)%hull.size()], p);
  }
  double totalArea = 0.0;
  for (int i = 0; i < hull.size(); i++) {
      totalArea += area(hull[i], hull[(i+1)%hull.size()], hull[(i+2)%hull.size()]);
  }
  totalArea = totalArea/2;
  // if (hullArea > totalArea+1e-5) {
  // std::cout<<"hull area "<<hullArea<<" "<<totalArea<<" "<<hullArea/totalArea<<" "<<std::endl;
  //   for(int i = 0; i < 4; i++) {
  //   std::cout<<i<<" "<<hull[i].x<<" "<<hull[i].y<<std::endl;
  //   }
  // }

  // 
  // double min_com_feet = 1e9;
  // Point feet_com = {fxy_avg[0], fxy_avg[1]};
  // for (int i = 0; i < hull.size(); i++) {
  //   double d = distanceFromPointToLine(hull[i], hull[(i+1)%hull.size()], feet_com);
  //   min_com_feet = std::min(min_com_feet, d);
  // }

  
  
  // // std::cout<<"com upper "<<com_position_upper[0]<<" "<<com_position_upper[1]<<" "<<com_position_upper[2]<<std::endl;
  // mju_addTo3(capture_point, com_position_upper);
  // mju_scl3(capture_point, capture_point, 0.5);
  // mju_copy(capture_point, com_position_upper, 3);
 

  // capture_point[0] -= kFallTime* parameters_[1];
  // mju_scl3(com_velocity_upper, {parameters_[1]}, 1);
  // com_velocity[0] -= parameters_[1];
  // double velocity_diff[3];
  // double desired_velocity[3] = {parameters_[1], 0, 0};
  // mju_sub3(velocity_diff, com_velocity, desired_velocity);
  // mju_addToScl3(capture_point, velocity_diff, kFallTime);
  // capture_point[0] += kFallTime * (com_velocity[0] - parameters_[1]);
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
  // mju_subFrom(fxy_avg, capture_point, 2);
  double balance;
  // double com_feet_distance = mju_norm(fxy_avg, 2);


  // Compute center of pressure

  double support_force = 0;
  support_force += mju_max(0, data->cfrc_ext[6*5+2]);
  support_force += mju_max(0, data->cfrc_ext[6*6+2]);
  support_force += mju_max(0, data->cfrc_ext[6*11+2]);
  support_force += mju_max(0, data->cfrc_ext[6*12+2]);


  double cop[2] = {0, 0};

  mju_addToScl(cop, data->xipos+3*5, mju_max(0, data->cfrc_ext[6*5+2]), 2);
  mju_addToScl(cop, data->xipos+3*6, mju_max(0, data->cfrc_ext[6*6+2]), 2);
  mju_addToScl(cop, data->xipos+3*11, mju_max(0, data->cfrc_ext[6*11+2]), 2);
  mju_addToScl(cop, data->xipos+3*12, mju_max(0, data->cfrc_ext[6*12+2]), 2);

  if (support_force > 0) {
    mju_scl(cop, cop, 1.0/support_force, 2);
  }
  else {
    // mju_scl(cop, fxy_avg, 1.0, 2);
    mju_copy(cop, capture_point, 2);
  }
  // mju_scl(cop, cop, 1.0/support_force, 2);
  // std::cout<<"cop "<<cop[0]<<" "<<cop[1]<<std::endl;

  


  double min_distance = 1e9;
  int min_idx = -1;
  for (int i = 0; i < hull.size(); i++) {
    double d = distanceFromPointToLine(hull[i], hull[(i+1)%hull.size()], p);
    min_distance = std::min(min_distance, d);
    if (min_distance == d) {
      min_idx = i;
    }
  }
  

  double com_feet_distance[2];
  mju_addTo(fxy_avg, cop, 2);
  mju_scl(fxy_avg, fxy_avg, 0.5, 2);
  mju_sub(com_feet_distance, fxy_avg, capture_point, 2);
  // mju_sub(com_feet_distance, cop, capture_point, 2);
  
  if (hullArea == totalArea) {
    // if (com_feet_distance<min_com_feet) {
      // balance = 0;
    // } else {
      // balance = com_feet_distance;
    
    balance = mju_norm(com_feet_distance, 2);
    // }
  } else {
    // p = {capture_point[0], capture_point[1]};
    
    
    // mju_sub(com_feet_distance, fxy_avg, capture_point, 2);
    hullArea = totalArea + area(hull[min_idx], hull[(min_idx+1)%hull.size()], p);
    // balance = com_feet_distance * hullArea/totalArea;
    balance = mju_norm(com_feet_distance, 2) * hullArea/totalArea;
    // * pow(hullArea/totalArea, 5);
    
  }
  residual[counter++] = mju_min(1000, balance);
  // residual[counter++] = (mju_exp(com_feet_distance)-1);
  // *mju_max(1, (hullArea/totalArea));
  // residual[counter++] = 100*(mju_max(1, hullArea/totalArea)-1);

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

  // double* hip_l_position = SensorByName(model, data, "hip_l");
  // double* hip_r_position = SensorByName(model, data, "hip_r");
  // double hip_dir[2];
  // mju_sub(hip_dir, hip_l_position, hip_r_position, 2);
  // // std::cout<<mju_norm(hip_dir, 2)<<std::endl;
  // mju_normalize(hip_dir, 2);

  // double left_feet_com[2];
  // mju_add(left_feet_com, f1_position, f2_position, 2);
  // mju_scl(left_feet_com, left_feet_com, 0.5, 2);
  // double right_feet_com[2];
  // mju_add(right_feet_com, f3_position, f4_position, 2);
  // mju_scl(right_feet_com, right_feet_com, 0.5, 2);


  // double feet_dir[2];
  // mju_sub(feet_dir, f1_position, f3_position, 2); //calcn position
  // // mju_normalize(feet_dir, 2);
  // double toe_dir[2];
  // mju_sub(toe_dir, f2_position, f4_position, 2);

  // double* knee_l_position = SensorByName(model, data, "knee_l");
  // double* knee_r_position = SensorByName(model, data, "knee_r");
  // double knee_dir[2];
  // mju_sub(knee_dir, knee_l_position, knee_r_position, 2);


  // double feet_cross = mju_min(0, mju_dot(hip_dir, feet_dir, 2)-0.2);
  // feet_cross += mju_min(0, mju_dot(hip_dir, toe_dir, 2)-0.2);
  // feet_cross += mju_min(0, mju_dot(hip_dir, knee_dir, 2)-0.15);
  // std::cout<<"dot product "<<mju_dot(hip_dir, feet_dir, 2)<<" "<<mju_dot(hip_dir, toe_dir, 2)<<" "<<mju_dot(hip_dir, knee_dir, 2)<<" "<<feet_cross<<std::endl;
  
  // double feet_dir_l[2];
  // mju_sub(feet_dir_l, left_feet_com, hip_l_position, 2);
  // double feet_dir_r[2];
  // mju_sub(feet_dir_r, right_feet_com, hip_r_position, 2);
  // double feet_cross = mju_min(0, mju_dot(hip_dir, feet_dir_l, 2));
  // feet_cross += mju_min(0, -mju_dot(hip_dir, feet_dir_r, 2));
  // std::cout<<"feet cross "<<mju_dot(hip_dir, feet_dir_l, 2)<<" "<<-mju_dot(hip_dir, feet_dir_r, 2)<<std::endl;
  // residual[counter++] = feet_cross;

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

  residual[counter++] = mju_min(0, (mju_dot(com_velocity, forward_dir, 2) - parameters_[1]));
  // *(mju_min(1, 1-upright))) * (mju_min(1, head_feet_error/parameters_[0])) * mju_min(1-balance, 1);
  // *(1 - mju_min(head_feet_error, parameters_[0])/parameters_[0]);

  // ----- forward velocity angle ----- //
  double cos_theta = mju_dot(com_velocity, forward_dir, 2) / (mju_norm(com_velocity, 3)+1e-7);
  double sin_theta = mju_sqrt(1 - cos_theta * cos_theta);
  double side_velocity = mju_norm(com_velocity, 2) * sin_theta;
  residual[counter++] = side_velocity;
  // residual[counter++] = mju_norm(data->qvel+4, 2);
  // std::cout<<"side velocity "<<side_velocity<<" "<<sin_theta<<" "<<mju_norm(com_velocity, 2)<<std::endl;
  
  // ----- forward ----- //
  // residual[counter++] = mju_dot(hip_dir, forward_dir, 2);

  double* torso_forward = SensorByName(model, data, "torso_forward");
  double* pelvis_forward = SensorByName(model, data, "pelvis_forward");
  double* knee_left_forward = SensorByName(model, data, "knee_left_forward");
  double* knee_right_forward = SensorByName(model, data, "knee_right_forward");

  double forward = 0;

  forward += 0 * (1- mju_dot(torso_forward, forward_dir, 3));
  forward += 1 * (1- mju_dot(pelvis_forward, forward_dir, 2));
  forward += 0 * (1- mju_dot(knee_left_forward, forward_dir, 2));
  forward += 0 * (1- mju_dot(knee_right_forward, forward_dir, 2));
  residual[counter++] = forward;

  // ---contact---- //
  // double contact_num = 0;
  double body_force = 0;
  int floor = mj_name2id(model, mjOBJ_GEOM, "floor");
  std::vector<double> include_body = {};
  for (int i = 0; i < data->ncon; i++) {
  // int cube = mj_name2id(model, mjOBJ_GEOM, "cube");
  
    mjContact *g = data->contact + i;
    if (g->geom1 != floor && g->geom2 != floor) {
      // contact_num += 1;
      int bodyid = model->geom_bodyid[g->geom1];
      int bodyid2 = model->geom_bodyid[g->geom2];
      if (bodyid > 12 || bodyid2 > 12) {
        continue;
      }
      if (std::find(include_body.begin(), include_body.end(), bodyid) == include_body.end()) {
        include_body.push_back(bodyid);
        
        body_force += mju_norm3(data->cfrc_ext+6*bodyid);
        // std::cout<<bodyid<<" "<<mj_id2name(model, mjOBJ_GEOM, g->geom1)<<" "<<mj_id2name(model, mjOBJ_BODY, bodyid)<<" "<<mju_norm3(data->cfrc_ext+6*bodyid)<<std::endl;
      }
      // std::cout<<"contact "<<mj_id2name(model, mjOBJ_GEOM, g->geom1)<<" "<<mj_id2name(model, mjOBJ_BODY, bodyid)<<" "<<mj_id2name(model, mjOBJ_GEOM, g->geom2)<<std::endl;
    }
  }
  // residual[counter++] = contact_num;
  
  // double body_force = 0;
  // for (int i = 0; i < 91; i++) {
  //   // std::cout<<data->qpos[i]<<" ";
  //   if (i != 5 && i != 6 && i != 11 && i != 12) {
  //     if (mju_norm3(data->cfrc_ext+6*i) > 1e-7) {
  //       std::cout<<mj_id2name(model, mjOBJ_BODY, i)<<" "<<data->cfrc_ext[6*i]<<" "<<data->cfrc_ext[6*i+1]<<" "<<data->cfrc_ext[6*i+2]<<std::endl;
  //     }
  //     body_force += mju_norm3(data->cfrc_ext+6*i);
  //   }
    
  // }
  residual[counter++] = mju_min(1e4, body_force);

  // ----- penalize large support force ----- //
  // double ground_force = 0;
  // ground_force += data->cfrc_ext[6*5 + 2];
  // ground_force += data->cfrc_ext[6*6 + 2];
  // ground_force += data->cfrc_ext[6*11 + 2];
  // ground_force += data->cfrc_ext[6*12 + 2];
  // residual[counter++] = mju_max(0, ground_force-1.2*mju_sum(model->body_mass, 91)*9.8);
  // ----- penalize large joint pos----- //
  std::vector<int> includeList = {6, 7, 8, 12, 15, 16, 17, 21, 22, 23, 27, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 55, 56, 58, 59, 60, 61, 62, 75, 76, 78, 79, 80, 81, 82}; 
  double joint_sum = 0;
  // int max_jnt = -1;
  // double max_jnt_force = -1;
  for (int i = 0; i < includeList.size(); i++) {
    double norm_pos = data->qpos[includeList[i]];
    // /(model->jnt_range[2*includeList[i]+1] - model->jnt_range[2*includeList[i]]);
    joint_sum += norm_pos * norm_pos;
    // double pos_vio = mju_max(model->jnt_range[2*includeList[i]] - data->qpos[includeList[i]],  data->qpos[includeList[i]] - model->jnt_range[2*includeList[i]+1]);
    // pos_vio = mju_max(0, pos_vio);
    // if (pos_vio > 0) {
    //   std::cout<<"joint constraint "<<mj_id2name(model, mjOBJ_JOINT, includeList[i])<<" "<<data->qfrc_constraint[includeList[i]]<<" "<<pos_vio<<std::endl;
    // }
    // joint_sum += pos_vio;
    // joint_sum += abs(data->qfrc_constraint[includeList[i]]);
    //  if (abs(data->qfrc_constraint[includeList[i]]/(model->jnt_range[2*includeList[i]+1] - model->jnt_range[2*includeList[i]])) > max_jnt_force) {
    // // if (abs(data->qfrc_constraint[includeList[i]]) > max_jnt_force) {
    //   max_jnt_force = abs(data->qfrc_constraint[includeList[i]]);
    //   max_jnt = includeList[i];
    // }
  }
  // // std::cout<<"jnt vio "<<joint_sum<<std::endl;

  // // std::cout<<"joint constraint "<<mj_id2name(model, mjOBJ_JOINT, max_jnt)<<" "<<data->qfrc_constraint[max_jnt]<<std::endl;
  residual[counter++] = joint_sum;


  // // ----- penalize large orthgonal force ----- //
  // double orth_force = 0;
  // double orth_dir[2] = {-forward_dir[1], forward_dir[0]};
  // for (int i = 0; i < 91; i++) {
  //     orth_force += mju_dot(data->cfrc_ext+6*i, orth_dir, 2);
  //   }
  
  // residual[counter++] = orth_force;

  // ----- match forward force ----- //
  double acc_sum = 0;
  for (int i = 0; i < 85; i++) {
    double acc = data->qacc[i];
    acc_sum += acc*acc;
  }
  residual[counter++] = acc_sum;
  double desire_velocity[3] = {parameters_[1], 0, 0};

  double deisre_force[3];
  mju_sub3(deisre_force, desire_velocity, com_velocity);
  mju_scl3(deisre_force, deisre_force, 1/model->opt.timestep);
  double mass = mju_sum(model->body_mass, 91);
  mju_scl3(deisre_force, deisre_force, mass);
  deisre_force[2] += 1.2 * mass * 9.8;
  double deisre_force_dir[2];
  mju_copy(deisre_force_dir, deisre_force, 2);
  mju_normalize(deisre_force_dir, 2);



  double feet_force[3] = {0, 0, 0};



  mju_addTo3(feet_force, data->cfrc_ext+6*5);
  mju_addTo3(feet_force, data->cfrc_ext+6*6);
  mju_addTo3(feet_force, data->cfrc_ext+6*11);
  mju_addTo3(feet_force, data->cfrc_ext+6*12);


  // mju_scl3(deisre_force_dir, deisre_force_dir, -1);
  // double horizontal_force_match = mju_dot(deisre_force_dir, feet_force, 2);
  double vertical_force_match = mju_max(0, feet_force[2] - deisre_force[2]);
  // mju_sub3(feet_force_match, deisre_force, feet_force);
  


  // double force_diff[3];
  // mju_sub3(force_diff, deisre_force, feet_force);
  // if (force_diff[2] > 0) {
  //   force_diff[2] = 0;
  // } 
  // force_diff[1] = 0;
  // std::cout<<"force diff "<<force_diff[0]<<" "<<force_diff[1]<<" "<<force_diff[2]<<std::endl;
  // residual[counter++] = mju_max(0, force_diff[0]);


  // residual[counter++] = mju_min(0, horizontal_force_match);
  residual[counter++] = vertical_force_match;

  //arm velocity matches leg velocity
  double left_arm_velocity[3] = {0, 0, 0};
  double right_arm_velocity[3] = {0, 0, 0};
  double left_leg_velocity[3] = {0, 0, 0};
  double right_leg_velocity[3] = {0, 0, 0};

  for (int i = 0; i < model->nbody; i++) {
    
    if (i>1 && i<=7) { //right leg
      // std::cout<<"velocity "<<data->cvel[6*i]<<" "<<data->cvel[6*i+1]<<" "<<data->cvel[6*i+2]<<std::endl;
      mju_addToScl3(right_leg_velocity, data->cvel+6*i+3, model->body_mass[i]);
      
    }
    // std::cout<<"velocity before"<<right_leg_velocity[0]<<" "<<right_leg_velocity[1]<<" "<<right_leg_velocity[2]<<std::endl;
    
    // std::cout<<"velocity scl"<<right_leg_velocity[0]<<" "<<right_leg_velocity[1]<<" "<<right_leg_velocity[2]<<" "<<mju_sum(model->body_mass+2, 6)<<std::endl;
    if (i>7 && i<=13) { //left leg
      mju_addToScl3(left_leg_velocity, data->cvel+6*i+3, model->body_mass[i]);
    }
    

    if (i>41 && i<=46) { //right arm
      mju_addToScl3(right_arm_velocity, data->cvel+6*i+3, model->body_mass[i]);
    }
    

    if (i>52 && i<=57) { //left arm
      mju_addToScl3(left_arm_velocity, data->cvel+6*i+3, model->body_mass[i]);
    }
    
  }
  mju_scl3(right_leg_velocity, right_leg_velocity, 1/mju_sum(model->body_mass+2, 6));
  mju_scl3(left_leg_velocity, left_leg_velocity, 1/mju_sum(model->body_mass+8, 6));
  mju_scl3(right_arm_velocity, right_arm_velocity, 1/mju_sum(model->body_mass+42, 5));
  mju_scl3(left_arm_velocity, left_arm_velocity, 1/mju_sum(model->body_mass+53, 5));
  // std::cout<<"velocity "<<left_arm_velocity[0]<<" "<<right_arm_velocity[0]<<" "<<left_leg_velocity[0]<<" "<<right_leg_velocity[0]<<std::endl;
  double arm_leg_diff1[3];
  mju_sub3(arm_leg_diff1, left_arm_velocity, right_leg_velocity);
  double arm_leg_diff2[3];
  mju_sub3(arm_leg_diff2, right_arm_velocity, left_leg_velocity);
  // std::cout<<"diff1 "<<arm_leg_diff1[0]<<" "<<arm_leg_diff1[1]<<" "<<arm_leg_diff1[2]<<std::endl;
  // std::cout<<"diff2 "<<arm_leg_diff2[0]<<" "<<arm_leg_diff2[1]<<" "<<arm_leg_diff2[2]<<std::endl;
  residual[counter++] = mju_norm(arm_leg_diff1, 3) + mju_norm(arm_leg_diff2, 3);

  // residual[counter++] = abs(vertical_force_match);
  //  + vertical_force_match * vertical_force_match;

  // double forward_force = 0;
  // double orth_dir[2] = {-forward_dir[1], forward_dir[0]};
  // for (int i = 0; i < 91; i++) {
  //     orth_force += mju_dot(data->cfrc_ext+6*i, orth_dir, 2);
  //   }
  
  // residual[counter++] = orth_force;

  // ----- penalize large head movement ----- //

  // residual[counter++] += mju_norm3(data->qvel + 42);

  // residual[counter++] = mju_max(1e4, body_force);
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

void Walk4::TransitionLocked(mjModel* model, mjData* data) {
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

  double head_height = SensorByName(model, data, "head_position")[2];
  if (head_height < 0.8) {
    mju_copy(data->qpos, model->qpos0, model->nq);
    mju_zero(data->qvel, model->nv);
    mju_zero(data->act, model->na);
  }

  double support_force = 0;

  support_force += mju_max(0, data->cfrc_ext[6*5+2]);
  support_force += mju_max(0, data->cfrc_ext[6*6+2]);
  support_force += mju_max(0, data->cfrc_ext[6*11+2]);
  support_force += mju_max(0, data->cfrc_ext[6*12+2]);

  double cop[2] = {0, 0};

  mju_addToScl(cop, data->xipos+3*5, mju_max(0, data->cfrc_ext[6*5+2]), 2);
  mju_addToScl(cop, data->xipos+3*6, mju_max(0, data->cfrc_ext[6*6+2]), 2);
  mju_addToScl(cop, data->xipos+3*11, mju_max(0, data->cfrc_ext[6*11+2]), 2);
  mju_addToScl(cop, data->xipos+3*12, mju_max(0, data->cfrc_ext[6*12+2]), 2);

  if (support_force > 0) {
    mju_scl(cop, cop, 1.0/support_force, 2);
    // if (abs(cop[0]) > 10) {
    //   std::cout<<"cop "<<cop[0]<<" "<<cop[1]<<" "<<data->cfrc_ext[6*5+2]<<" "<<data->cfrc_ext[6*6+2]<<" "<<data->cfrc_ext[6*11+2]<<" "<<data->cfrc_ext[6*12+2]<<std::endl;
    //   std::cout<<"xpos "<<data->xipos[3*5]<<" "<<data->xipos[3*5+1]<<" "<<data->xipos[3*6]<<" "<<data->xipos[3*6+1]<<" "<<data->xipos[3*11]<<" "<<data->xipos[3*11+1]<<" "<<data->xipos[3*12]<<" "<<data->xipos[3*12+1]<<std::endl;
    // }
    mju_copy(data->mocap_pos, cop, 2);
    // std::cout<<"copy "<<cop[0]<<" "<<cop[1]<<" "<<data->cfrc_ext[6*5+2]<<" "<<data->cfrc_ext[6*6+2]<<" "<<data->cfrc_ext[6*11+2]<<" "<<data->cfrc_ext[6*12+2]<<std::endl;
  }
  // else {
  //   // mju_scl(cop, fxy_avg, 1.0, 2);
  //   mju_copy(cop, capture_point, 2);
  // }



  if (model->nv >= 170) {
    mju_zero(data->qvel+85, 85);
  }
  
  // std::cout<<model->nuserdata<<std::endl;
}

}  // namespace mjpc
