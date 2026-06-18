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

#include "mjpc/tasks/musculoskeletal/tracking/tracking.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <string>
#include <tuple>

#include <mujoco/mujoco.h>
#include "mjpc/utilities.h"


namespace mjpc::musculoskeletal {

std::string Tracking::XmlPath() const {
  return GetModelPath("musculoskeletal/tracking/task.xml");
}
std::string Tracking::Name() const { return "MS Track"; }

// ------------- Residuals for humanoid tracking task -------------
//   Number of residuals:
//     Residual (0): Joint vel: minimise joint velocity
//     Residual (1): Control: minimise control
//     Residual (2-11): Tracking position: minimise tracking position error
//         for {root, head, toe, heel, knee, hand, elbow, shoulder, hip}.
//     Residual (11-20): Tracking velocity: minimise tracking velocity error
//         for {root, head, toe, heel, knee, hand, elbow, shoulder, hip}.
//   Number of parameters: 0
// ----------------------------------------------------------------
void Tracking::ResidualFn::Residual(const mjModel *model, const mjData *data,
                                  double *residual) const {

  // get indexs = data->time/0.002
  int current_index = floor(data->time/0.002);
  // get keyqpos
  // mjtNum key_qpos[1000];
  // mju_copy(key_qpos, model->key_qpos + model->nq * current_index, model->nq);
  // //get current qpos
  // mjtNum current_qpos[1000];
  // mju_copy(current_qpos, data->qpos, model->nq);
  double* pos_diff = new double[model->nq];
  mju_sub(pos_diff, data->qpos, model->key_qpos + current_index * model->nq, model->nq);
  double* vel_diff = new double[model->nq];
  mju_sub(vel_diff, data->qvel, model->key_qvel + current_index * model->nq, model->nq);

  //compute pos diff
  // mjtNum pos_diff[1000];
  // mju_sub(pos_diff, current_qpos, key_qpos, model->nq);
  // ----- residual ----- //
  int counter = 0;
  int indexs[37] = {0,1,2,3,4,5,6, 7, 8, 12, 15, 16, 21, 22, 23, 27, 30,31, 55, 56, 58, 59, 60, 75, 76, 78, 79, 80, 36, 37, 38, 39, 40, 41, 42, 43, 44};
  // we only compare the upper array indexs
  for (counter = 0; counter < 37; counter++) {
    // printf("pos_diff[%d] = %f\n", indexs[i], pos_diff[indexs[i]]);
    mju_copy(residual + counter,pos_diff+indexs[counter],1);
  }

  for (counter = 37; counter < 74; counter++) {
    // printf("pos_diff[%d] = %f\n", indexs[i], pos_diff[indexs[i]]);
    mju_copy(residual + counter,pos_diff+indexs[counter-37],1);
  }
  // printf("counter = %d\n", counter);
  // counter += 37;
  // ----- get mocap frames ----- //
  // get motion start index
  // int start = MotionStartIndex(current_mode_);
  // // get motion trajectory length
  // int length = MotionLength(current_mode_);
  // double current_index = (data->time - reference_time_) * kFps + start;
  // int last_key_index = start + length - 1;

  // Positions:
  // We interpolate linearly between two consecutive key frames in order to
  // provide smoother signal for tracking.
  // int key_index_0, key_index_1;
  // double weight_0, weight_1;
  // std::tie(key_index_0, key_index_1, weight_0, weight_1) =
  //     ComputeInterpolationValues(current_index, last_key_index);



  // // ----- joint velocity ----- //
  // mju_copy(residual + counter, data->qvel + 6, model->nv - 6);
  // counter += model->nv - 6;

  // // ----- action ----- //
  // mju_copy(&residual[counter], data->ctrl, model->nu);
  // counter += model->nu;

  // // ----- position ----- //
  // // Compute interpolated frame.
  // //   auto get_body_qpos = [&](const std::string &body_name, double result[3]) {
  // //   std::string mocap_body_name = "mocap[" + body_name + "]";
  // //   int mocap_body_id = mj_name2id(model, mjOBJ_BODY, mocap_body_name.c_str());
  // //   assert(0 <= mocap_body_id);
  // //   int body_mocapid = model->body_mocapid[mocap_body_id];
  // //   assert(0 <= body_mocapid);

  // //   // current frame
  // //   mju_scl3(
  // //       result,
  // //       model->key_mpos + model->nmocap * 3 * key_index_0 + 3 * body_mocapid,
  // //       weight_0);

  // //   // next frame
  // //   mju_addToScl3(
  // //       result,
  // //       model->key_mpos + model->nmocap * 3 * key_index_1 + 3 * body_mocapid,
  // //       weight_1);
  // // };

  // auto get_body_mpos = [&](const std::string &body_name, double result[3]) {
  //   std::string mocap_body_name = "mocap[" + body_name + "]";
  //   int mocap_body_id = mj_name2id(model, mjOBJ_BODY, mocap_body_name.c_str());
  //   assert(0 <= mocap_body_id);
  //   int body_mocapid = model->body_mocapid[mocap_body_id];
  //   assert(0 <= body_mocapid);

  //   // current frame
  //   mju_scl3(
  //       result,
  //       model->key_mpos + model->nmocap * 3 * key_index_0 + 3 * body_mocapid,
  //       weight_0);

  //   // next frame
  //   mju_addToScl3(
  //       result,
  //       model->key_mpos + model->nmocap * 3 * key_index_1 + 3 * body_mocapid,
  //       weight_1);
  // };

  // auto get_body_sensor_pos = [&](const std::string &body_name,
  //                                double result[3]) {
  //   std::string pos_sensor_name = "tracking_pos[" + body_name + "]";
  //   double *sensor_pos = SensorByName(model, data, pos_sensor_name.c_str());
  //   mju_copy3(result, sensor_pos);
  // };

  // // compute marker and sensor averages
  // double avg_mpos[3] = {0};
  // double avg_sensor_pos[3] = {0};
  // int num_body = 0;
  // for (const auto &body_name : body_names) {
  //   double body_mpos[3];
  //   double body_sensor_pos[3];
  //   get_body_mpos(body_name, body_mpos);
  //   mju_addTo3(avg_mpos, body_mpos);
  //   get_body_sensor_pos(body_name, body_sensor_pos);
  //   mju_addTo3(avg_sensor_pos, body_sensor_pos);
  //   num_body++;
  // }
  // mju_scl3(avg_mpos, avg_mpos, 1.0/num_body);
  // mju_scl3(avg_sensor_pos, avg_sensor_pos, 1.0/num_body);

  // // residual for averages
  // mju_sub3(&residual[counter], avg_mpos, avg_sensor_pos);
  // counter += 3;

  // for (const auto &body_name : body_names) {
  //   double body_mpos[3];
  //   get_body_mpos(body_name, body_mpos);

  //   // current position
  //   double body_sensor_pos[3];
  //   get_body_sensor_pos(body_name, body_sensor_pos);

  //   mju_subFrom3(body_mpos, avg_mpos);
  //   mju_subFrom3(body_sensor_pos, avg_sensor_pos);

  //   mju_sub3(&residual[counter], body_mpos, body_sensor_pos);

  //   counter += 3;
  // }

  // // ----- velocity ----- //
  // for (const auto &body_name : body_names) {
  //   std::string mocap_body_name = "mocap[" + body_name + "]";
  //   std::string linvel_sensor_name = "tracking_linvel[" + body_name + "]";
  //   int mocap_body_id = mj_name2id(model, mjOBJ_BODY, mocap_body_name.c_str());
  //   assert(0 <= mocap_body_id);
  //   int body_mocapid = model->body_mocapid[mocap_body_id];
  //   assert(0 <= body_mocapid);

  //   // compute finite-difference velocity
  //   mju_copy3(
  //       &residual[counter],
  //       model->key_mpos + model->nmocap * 3 * key_index_1 + 3 * body_mocapid);
  //   mju_subFrom3(
  //       &residual[counter],
  //       model->key_mpos + model->nmocap * 3 * key_index_0 + 3 * body_mocapid);
  //   mju_scl3(&residual[counter], &residual[counter], kFps);

  //   // subtract current velocity
  //   double *sensor_linvel =
  //       SensorByName(model, data, linvel_sensor_name.c_str());
  //   mju_subFrom3(&residual[counter], sensor_linvel);

  //   counter += 3;
  // }


  CheckSensorDim(model, counter);
}

// --------------------- Transition for humanoid task -------------------------
//   Set `data->mocap_pos` based on `data->time` to move the mocap sites.
//   Linearly interpolate between two consecutive key frames in order to
//   smooth the transitions between keyframes.
// ----------------------------------------------------------------------------
void Tracking::TransitionLocked(mjModel *model, mjData *d) {

  // std::cout<<d->time<<std::endl;
  if (d->time == 0) {
    mju_copy(d->qpos, model->key_qpos, model->nq);
  }
  // // get motion start index
  // int start = MotionStartIndex(mode);
  // // get motion trajectory length
  // int length = MotionLength(mode);

  // // check for motion switch
  // if (residual_.current_mode_ != mode || d->time == 0.0) {
  //   residual_.current_mode_ = mode;       // set motion id
  //   residual_.reference_time_ = d->time;  // set reference time

  //   // set initial state
  //   mju_copy(d->qpos, model->key_qpos + model->nq * start, model->nq);
  //   mju_copy(d->qvel, model->key_qvel + model->nv * start, model->nv);
  // }

  // // indices
  // double current_index = (d->time - residual_.reference_time_) * kFps + start;
  // int last_key_index = start + length - 1;

  // // Positions:
  // // We interpolate linearly between two consecutive key frames in order to
  // // provide smoother signal for tracking.
  // int key_index_0, key_index_1;
  // double weight_0, weight_1;
  // std::tie(key_index_0, key_index_1, weight_0, weight_1) =
  //     ComputeInterpolationValues(current_index, last_key_index);

  // mj_markStack(d);

  // mjtNum *mocap_pos_0 = mj_stackAllocNum(d, 3 * model->nmocap);
  // mjtNum *mocap_pos_1 = mj_stackAllocNum(d, 3 * model->nmocap);
  
  // // Compute interpolated frame.
  // mju_scl(mocap_pos_0, model->key_mpos + model->nmocap * 3 * key_index_0,
  //         weight_0, model->nmocap * 3);

  // mju_scl(mocap_pos_1, model->key_mpos + model->nmocap * 3 * key_index_1,
  //         weight_1, model->nmocap * 3);

  // mju_copy(d->mocap_pos, mocap_pos_0, model->nmocap * 3);
  // mju_addTo(d->mocap_pos, mocap_pos_1, model->nmocap * 3);

  // mj_freeStack(d);
}

}  // namespace mjpc::humanoid
