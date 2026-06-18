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

#ifndef MJPC_PLANNERS_HIERARHICAL_SAMPLING_POLICY_H_
#define MJPC_PLANNERS_HIERARHICAL_SAMPLING_POLICY_H_
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
class HierarchicalSamplingPolicy : public Policy {
 public:
  // constructor
  HierarchicalSamplingPolicy() = default;

  // destructor
  ~HierarchicalSamplingPolicy() override = default;

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

  // Solve a quadratic program
  Eigen::VectorXd solve_qp(
    Eigen::SparseMatrix<double> hessian, 
    Eigen::VectorXd gradient, 
    Eigen::SparseMatrix<double> constraintMatrix,
    Eigen::VectorXd lowerBound, 
    Eigen::VectorXd upperBound, 
    Eigen::VectorXd initialGuess
    ) const;
  
  //solve a linear program
  Eigen::VectorXd linprog(Eigen::VectorXd c,
                      Eigen::MatrixXd A,
                      Eigen::VectorXd b,
                      Eigen::VectorXd x0) const;


  // get qfrc
  double* get_qfrc(mjModel* model, double* target_qpos) const;

  // get control
  Eigen::VectorXd get_ctrl(double* target_pos, double* qfrc) const;

  // copy policy
  void CopyFrom(const HierarchicalSamplingPolicy& policy, int horizon);

  // copy parameters
  void SetPlan(const mjpc::spline::TimeSpline& plan);


  // ----- members ----- //
  const mjModel* model;
  mjpc::spline::TimeSpline plan;
  int num_spline_points;
  mjData* data_copy; //for inverse dynamics
  mjData* data_copy2; // for control compute
  mjModel* model_copy;

  int dim_high_level_action;  // number of high-level actions  
  std::vector<double> high_level_actions;   // (horizon-1 x num_action)

  //used in QP
  Eigen::SparseMatrix<double> P;
  Eigen::VectorXd q;


  //used in LP
  Eigen::VectorXd c;
  // Eigen::MatrixXd linearMatrix;

  // Eigen::SparseMatrix<double> linearMatrix;

  // bool initialized = false;
};

}  // namespace mjpc

#endif  // MJPC_PLANNERS_SAMPLING_POLICY_H_
