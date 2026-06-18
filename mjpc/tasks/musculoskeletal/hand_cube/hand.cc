// Copyright 2024 DeepMind Technologies Limited
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

#include "mjpc/tasks/musculoskeletal/hand_cube/hand.h"
#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/utilities.h"

namespace mjpc {
  namespace musculoskeletal {
std::string HandCube::XmlPath() const {
  return GetModelPath("musculoskeletal/hand_cube/task.xml");
}
std::string HandCube::Name() const { return "MS HandCube"; }
void HandCube::ResidualFn::Residual(const mjModel *model, const mjData *data,
                                   double *residual) const {
  int counter = 0;

  // ---------- Cube position (must keep this term) ----------
  double *cube_position = SensorByName(model, data, "cube_position");
  double cube_goal_position[3] = {0.25, -0.2, 1.1};
  mju_sub3(residual + counter, cube_position, cube_goal_position);
  counter += 3;


  // Sanity check
  CheckSensorDim(model, counter);
}

void HandCube::TransitionLocked(mjModel *model, mjData *data) {
  // Check for contact between the cube and the floor
  int cube = mj_name2id(model, mjOBJ_GEOM, "cube");
  int floor = mj_name2id(model, mjOBJ_GEOM, "floor");

  bool on_floor = false;
  for (int i = 0; i < data->ncon; i++) {
    mjContact *g = data->contact + i;
    if ((g->geom1 == cube && g->geom2 == floor) ||
        (g->geom2 == cube && g->geom1 == floor)) {
      on_floor = true;
      break;
    }
  }

  // If the cube is on the floor and not moving, reset it
  // double *cube_lin_vel = SensorByName(model, data, "cube_linear_velocity");
  if (
    // (on_floor && mju_norm3(cube_lin_vel) < 0.001) or 
  data->time == 0
  ) {
    // int cube_body = mj_name2id(model, mjOBJ_BODY, "cube");
    // if (cube_body != -1) {
    //   int jnt_qposadr = model->jnt_qposadr[model->body_jntadr[cube_body]];
    //   int jnt_veladr = model->jnt_dofadr[model->body_jntadr[cube_body]];
    //   mju_copy(data->qpos + jnt_qposadr, model->key_qpos + jnt_qposadr, 7);
    //   mju_zero(data->qvel + jnt_veladr, 6);
    // }
    mju_copy(data->qpos , model->key_qpos, model->nq);
    mju_zero(data->qvel, model->nv);

    // Step the simulation forward
    mutex_.unlock();
    mj_forward(model, data);
    mutex_.lock();
  }

  // if (data->time > change_dir_time) {
    
  //   // std::cout<<cube_position[2]<<" "<<cube_goal_position[2]<<std::endl;
  //   // randomly set the goal position
  //   // double *cube_goal_position = SensorByName(model, data, "cube_goal_position");
  //   data->qpos[44] = rand() % 2*3.14 - 3.14;
  //   data->qpos[45] = rand() % 2*3.14 - 3.14;
  //   data->qpos[46] = rand() % 2*3.14 - 3.14;
  //   change_dir_time += 5;
  //   mju_zero(data->qvel+44, 3);
  //   // cube_goal_position[0] = 0.25 + 0.1 * (rand() % 10);
  //   // cube_goal_position[1] = -0.2 + 0.1 * (rand() % 10);
  //   // cube_goal_position[2] = 1.1 + 0.1 * (rand() % 10);
  // }

  
}
  }  // namespace musculoskeletal
}  // namespace mjpc
