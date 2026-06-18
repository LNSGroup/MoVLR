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

#ifndef MJPC_PLANNERS_HIERARHICAL_PD_POLICY_H_
#define MJPC_PLANNERS_HIERARHICAL_PD_POLICY_H_
// osqp-eigen
#include <OsqpEigen/OsqpEigen.h>
// eigen
#include <Eigen/Dense>
#include <mujoco/mujoco.h>
#include "mjpc/planners/policy.h"
#include "mjpc/spline/spline.h"
#include "mjpc/task.h"
// #include <glpk.h>

namespace mjpc {

// policy for sampling planner
class HierarchicalPDPolicy : public Policy {
 public:
  // constructor
  HierarchicalPDPolicy() = default;

  // destructor
  ~HierarchicalPDPolicy() override = default;

  // ----- methods ----- //

  // allocate memory
  void Allocate(const mjModel* model, const Task& task, int horizon) override;

  // reset memory to zeros
  void Reset(int horizon,
             const double* initial_repeated_action = nullptr) override;

  // set action from policy
  void Action(double* action, const double* state, double time) const override;
  
  void HierarchicalAction(double* action, mjData* data) const;

  // set action from higher level policy
  void HighToLowAction(double* high_level_action,  double* action, mjData* data) const;


  // get qfrc
  double* get_qfrc(mjModel* model, double* target_qpos) const;

  // get control
  //for target mjdata
  Eigen::VectorXd get_mus_ctrl() const;

  Eigen::VectorXd get_line_ctrl() const;
  //by pos
  Eigen::VectorXd get_ctrl(double* target_pos) const;
  //by vel
  Eigen::VectorXd get_ctrl2(double* target_qvel) const;
  // copy policy
  void CopyFrom(const HierarchicalPDPolicy& policy, int horizon);

  // copy parameters
  void SetPlan(const mjpc::spline::TimeSpline& plan);


  // ----- members ----- //
  const mjModel* model;
  mjpc::spline::TimeSpline plan;
  int num_spline_points;
  mjData* data_copy; //for inverse dynamics
  mjData* data_copy2; // for control compute

  int dim_high_level_action;  // number of high-level actions  
  std::vector<double> high_level_actions;   // (horizon-1 x num_action)

  // std::vector<int> excludeList = {0, 1, 2, 3, 4, 5, 9, 10, 11, 13, 14, 18, 19, 20, 24, 25, 26, 28, 29, 33, 34, 35, 
  // // 36, 37, 38, 39, 40, 41, 42, 43, 44, 
  // 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 57, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 77, 83, 84};

  // std::vector<int> includeList = {6, 7, 8, 12, 15, 16, 17, 21, 22, 23, 27, 30, 31, 32, 
  // 36, 37, 38, 39, 40, 41, 42, 43, 44, 
  // 55, 56, 58, 59, 60, 61, 62, 75, 76, 78, 79, 80, 81, 82}; 

  // std::vector<int> spinalList = {
  //   36, 37, 38, 39, 40, 41, 42, 43, 44
  // };


  //ostrich
  // std::vector<int> excludeList = {0, 1, 2, 3, 4, 5, 22, 23, 26, 27, 30, 31, 34, 35, 38, 39, 42, 43, 46, 47, 50, 51, 54, 55};
  // std::vector<int> includeList = {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 28, 29, 32, 33, 36, 37, 40, 41, 44, 45, 48, 49, 52, 53}; 


  //mice
  // std::vector<int> excludeList = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 
  // // 82, 83, 
  // 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 
  // // 112, 113, 
  // 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 182, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309};
  // // std::vector<int> includeList = {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48}; 
  // std::vector<int> includeList = {77, 78, 79, 80, 81, 
  // 82, 83,
  // 107, 108, 109, 110, 111, 
  // 112, 113,
  // 179, 180, 181, 183, 184, 185, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 230, 231, 232, 233, 234, 235, 236}; //new mice
  // hand
  
  // std::vector<int> excludeList = {42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52};
  // std::vector<int> includeList = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41}; 
  std::vector<int> excludeList;
  std::vector<int> includeList;

  // std::vector<int> spinalList = {
  //   36, 37, 38, 39, 40, 41, 42, 43, 44
  // };

  
  

  // double high_level_action_copy[85];
  // std::cout<<"copy key "<<model->key_qpos[6]<<std::endl;
  // bool initialized = false;
};

}  // namespace mjpc

#endif  // MJPC_PLANNERS_SAMPLING_POLICY_H_
