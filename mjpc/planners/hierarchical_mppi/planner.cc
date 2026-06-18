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

#include "mjpc/planners/hierarchical_mppi/planner.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <shared_mutex>

#include <absl/random/random.h>
#include <absl/types/span.h>
#include <mujoco/mujoco.h>
#include "mjpc/array_safety.h"
#include "mjpc/planners/planner.h"
#include "mjpc/planners/sampling/planner.h"
#include "mjpc/spline/spline.h"
#include "mjpc/states/state.h"
#include "mjpc/task.h"
#include "mjpc/threadpool.h"
#include "mjpc/trajectory.h"
#include "mjpc/utilities.h"

#include <Eigen/Dense>

namespace mjpc {

namespace mju = ::mujoco::util_mjpc;
using mjpc::spline::TimeSpline;

// initialize data and settings
void HierarchicalMPPIPlanner::Initialize(mjModel* model, const Task& task) {
  // delete mjData instances since model might have changed.
  data_.clear();

  // allocate one mjData for nominal.
  ResizeMjData(model, 1);

  // model
  this->model = model;

  // task
  this->task = &task;

  // sampling noise
  std_initial_ =
      GetNumberOrDefault(0.05, model,
                         "sampling_exploration");        // initial variance
  std_min_ = GetNumberOrDefault(0.05, model, "std_min");  // minimum variance

  // set number of trajectories to rollout
  num_trajectory_ = GetNumberOrDefault(64, model, "sampling_trajectories");
  perturb_std = GetNumberOrDefault(0.0, model, "perturb_std");
  std::cout<<"perturb_std "<<perturb_std<<std::endl;

  // set number of elite samples max(best 10%, 2)
  n_elite_ =
      GetNumberOrDefault(std::max(num_trajectory_ / 10, 2), model, "n_elite");

  if (num_trajectory_ > kMaxTrajectory) {
    mju_error_i("Too many trajectories, %d is the maximum allowed.",
                kMaxTrajectory);
  }
}

// allocate memory
void HierarchicalMPPIPlanner::Allocate() {
  // initial state
  int num_state = model->nq + model->nv + model->na + 2*model->nu + 6*model->nbody;

  // state
  // for MS model, there is no mocap or userdata 
  state.resize(num_state);
  mocap.resize(7 * model->nmocap);
  userdata.resize(model->nuserdata);

  // mju_error_i(
  //       "test stop here", 0);
  // policy
  // int num_max_parameter = model->nu * kMaxTrajectoryHorizon;
  
  policy.Allocate(model, *task, kMaxTrajectoryHorizon);
  resampled_policy.Allocate(model, *task, kMaxTrajectoryHorizon);
  previous_policy.Allocate(model, *task, kMaxTrajectoryHorizon);

  int num_max_parameter = policy.dim_high_level_action * kMaxTrajectoryHorizon;
  // scratch
  parameters_scratch.resize(num_max_parameter);
  times_scratch.resize(kMaxTrajectoryHorizon);

  // noise
  // noise.resize(kMaxTrajectory * (model->nu * kMaxTrajectoryHorizon));
  noise.resize(kMaxTrajectory * (policy.dim_high_level_action * kMaxTrajectoryHorizon));

  // variance
  // variance.resize(model->nu * kMaxTrajectoryHorizon);  // (nu * horizon)
  variance.resize(policy.dim_high_level_action * kMaxTrajectoryHorizon);  // (nu * horizon)

  // need to initialize an arbitrary order of the trajectories
  trajectory_order.resize(kMaxTrajectory);
  for (int i = 0; i < kMaxTrajectory; i++) {
    trajectory_order[i] = i;
  }

  // trajectories and parameters
  for (int i = 0; i < kMaxTrajectory; i++) {
    trajectory[i].Initialize(num_state, model->nu, task->num_residual,
                             task->num_trace, kMaxTrajectoryHorizon);
    trajectory[i].Allocate(kMaxTrajectoryHorizon);
    candidate_policy[i].Allocate(model, *task, kMaxTrajectoryHorizon);
  }
  nominal_trajectory.Initialize(num_state, model->nu, task->num_residual,
                                task->num_trace, kMaxTrajectoryHorizon);
  nominal_trajectory.Allocate(kMaxTrajectoryHorizon);


  // allocate memory for state buffer and policy buffer
  state_buffer.resize(0);
  policy_buffer.resize(0);
  mirror_state_buffer.resize(0);
  mirror_policy_buffer.resize(0);
  cost_buffer.resize(0);

  best_state_buffer.resize(buffer_size*2);
  best_policy_buffer.resize(buffer_size*2);
  best_cost_buffer.resize(buffer_size);
  // for (int i = 0; i < buffer_size; i++) {
  //   policy_buffer[i].resize(policy.dim_high_level_action * kMaxTrajectoryHorizon);
  // }



}



// reset memory to zeros
void HierarchicalMPPIPlanner::Reset(int horizon,
                                const double* initial_repeated_action) {
  // state
  std::fill(state.begin(), state.end(), 0.0);
  std::fill(mocap.begin(), mocap.end(), 0.0);
  std::fill(userdata.begin(), userdata.end(), 0.0);
  time = 0.0;

  // policy parameters
  policy.Reset(horizon, initial_repeated_action);
  resampled_policy.Reset(horizon, initial_repeated_action);
  previous_policy.Reset(horizon, initial_repeated_action);

  // scratch
  std::fill(parameters_scratch.begin(), parameters_scratch.end(), 0.0);
  std::fill(times_scratch.begin(), times_scratch.end(), 0.0);

  // noise
  std::fill(noise.begin(), noise.end(), 0.0);

  // variance
  double var = std_initial_ * std_initial_;
  std::fill(variance.begin(), variance.end(), var);

  // trajectory samples
  for (int i = 0; i < kMaxTrajectory; i++) {
    trajectory[i].Reset(kMaxTrajectoryHorizon);
    candidate_policy[i].Reset(horizon);
  }
  nominal_trajectory.Reset(kMaxTrajectoryHorizon);

  for (const auto& d : data_) {
    mju_zero(d->ctrl, model->nu);
  }

  // improvement
  improvement = 0.0;

  // // clear state buffer and policy buffer
  // for (int i = 0; i < buffer_size; i++) {
  //   std::fill(state_buffer[i].begin(), state_buffer[i].end(), 0.0);
  //   std::fill(policy_buffer[i].begin(), policy_buffer[i].end(), 0.0);
  // }
  state_buffer.resize(0);
  policy_buffer.resize(0);
  mirror_state_buffer.resize(0);
  mirror_policy_buffer.resize(0);
  cost_buffer.resize(0);

  best_state_buffer.resize(buffer_size*2);
  best_policy_buffer.resize(buffer_size*2);
  best_cost_buffer.resize(buffer_size);

  std::fill(best_cost_buffer.begin(), best_cost_buffer.end(), 1.0e6);
}

// set state
void HierarchicalMPPIPlanner::SetState(const State& state) {
  state.CopyTo(this->state.data(), this->mocap.data(), this->userdata.data(),
               &this->time);
}

// optimize nominal policy using random sampling
void HierarchicalMPPIPlanner::OptimizePolicy(int horizon, ThreadPool& pool) {
  resampled_policy.plan.SetInterpolation(interpolation_);

  // if num_trajectory_ has changed, use it in this new iteration.
  // num_trajectory_ might change while this function runs. Keep it constant
  // for the duration of this function.
  int num_trajectory = num_trajectory_;

  // n_elite_ might change in the GUI - keep constant for in this function
  n_elite_ = std::min(n_elite_, num_trajectory);
  int n_elite = std::min(n_elite_, num_trajectory);

  // resize number of mjData
  ResizeMjData(model, pool.NumThreads());

  // copy nominal policy
  {
    const std::shared_lock<std::shared_mutex> lock(mtx_);
    resampled_policy.CopyFrom(policy, policy.num_spline_points);
  }

  // resample nominal policy to current time
  // std::cout<<"ResamplePolicy"<<std::endl;
  this->ResamplePolicy(horizon);

  // ----- rollout noisy policies ----- //
  // start timer
  auto rollouts_start = std::chrono::steady_clock::now();
  // std::cout<<"Rollouts"<<std::endl;
  // simulate noisy policies
  this->Rollouts(num_trajectory, horizon, pool);
  
  // sort candidate policies and trajectories by score
  for (int i = 0; i < num_trajectory; i++) {
    trajectory_order[i] = i;
  }

  // sort so that the first ncandidates elements are the best candidates, and
  // the rest are in an unspecified order
  std::partial_sort(
      trajectory_order.begin(), trajectory_order.begin() + num_trajectory,
      trajectory_order.begin() + num_trajectory,
      [&trajectory = trajectory](int a, int b) {
        return trajectory[a].total_return < trajectory[b].total_return;
      });
  // for (int i = 0; i < num_trajectory_; i++) {
  //   printf("total return of trajectory %d %f\n", i, trajectory[i].total_return);
  // }
  // stop timer
  rollouts_compute_time = GetDuration(rollouts_start);

  // ----- update policy ----- //
  // start timer
  auto policy_update_start = std::chrono::steady_clock::now();

  // dimensions
  int num_spline_points = resampled_policy.num_spline_points;
  // int num_parameters = num_spline_points * model->nu;
  // int num_parameters = num_spline_points * policy.dim_high_level_action;

  // averaged return over elites
  double avg_return = 0.0;

  // reset parameters scratch
  std::fill(parameters_scratch.begin(), parameters_scratch.end(), 0.0);

  //loop over elites to compute MPPI weights
  Eigen::VectorXd weights(n_elite);
  double weight_sum = 0.0;
  double min_ret = 1.0e6;


  // for (int i = 0; i < num_trajectory; i++) {
  //   int idx = i;
  //   std::cout<<"total return of trajectory "<<idx<<" "<<trajectory[idx].total_return<<std::endl;
  // }
  // trajectory_order[0] = 0;
  mppi_top1 += 1;
  mppi_top += n_elite;
  for (int i = 0; i < n_elite; i++) {
    int idx = trajectory_order[i];
    min_ret = mju_min(min_ret, trajectory[idx].total_return);
  }
  // std::cout<<"ref return "<<trajectory[0].total_return<<std::endl;
  for (int i = 0; i < n_elite; i++) {
    int idx = trajectory_order[i];
    weights(i) = exp(-1 / temp*(trajectory[idx].total_return - min_ret));
    // if (i == 0) {
    //   weights(i) = 1;
    // }
    // else {
    //   weights(i) = 0;
    // }
    if (idx == 0) {
      top_nominal += 1;
      if (i == 0) {
        top1_nominal += 1;
      }
      // std::cout<<"use nominal!"<<i<<" "<<std::endl;
    }
    else if (idx == 1 && state_buffer.size() >= buffer_size) {
      top_ref += 1;
      if (i == 0) {
        top1_ref += 1;
      }
      // std::cout<<"use ref!"<<i<<" "<<weights(i)<<" "<<state_buffer.size()<<std::endl;
    }
    // else if (idx == 2 && state_buffer.size() >= buffer_size) {
    //   top_ref_next += 1;
    //   if (i == 0) {
    //     top1_ref_next += 1;
    //   }
    //   // std::cout<<"use ref!"<<i<<" "<<weights(i)<<" "<<state_buffer.size()<<std::endl;
    // }
    else if (idx <current_rollout_num ) {
      top_current += 1;
      if (i == 0) {
        top1_current += 1;
      }
      // std::cout<<"use current!"<<i<<" "<<weights(i)<<std::endl;
    }
    // else if (idx < 10) {
    //   std::cout<<"use current pos!"<<std::endl;
    // }
    weight_sum += weights(i);
    // std::cout<<"weight: "<<i<<" "<<idx<<" "<<weights(i)<<" "<<trajectory[idx].total_return<<std::endl;
  }

  std::cout<<"mppi "<<mppi_top1<<" "<<mppi_top<<std::endl;
  std::cout<<"nominal "<<top1_nominal/mppi_top1<<" "<<top1_nominal/mppi_top<<std::endl;
  std::cout<<"ref "<<top1_ref/mppi_top1<<" "<<top1_ref/mppi_top<<std::endl;
  // std::cout<<"ref next"<<top1_ref_next/mppi_top1<<" "<<top1_ref_next/mppi_top<<std::endl;
  std::cout<<"current "<<top1_current/mppi_top1<<" "<<top1_current/mppi_top<<std::endl;
  
  // std::cout<<"weight_sum: "<<weights<<" "<<weight_sum<<std::endl;
  // loop over elites to compute average
  for (int i = 0; i < n_elite; i++) {
    // ordered trajectory index
    int idx = trajectory_order[i];

    // add parameters
    for (int t = 0; t < num_spline_points; t++) {
      TimeSpline::Node n = candidate_policy[idx].plan.NodeAt(t);
      // for (int j = 0; j < model->nu; j++) {
      //   parameters_scratch[i * model->nu + j] += n.values()[j];
      // }
      for (int j = 0; j < policy.dim_high_level_action; j++) {

        // parameters_scratch[i * policy.dim_high_level_action + j] += n.values()[j];
        parameters_scratch[t * policy.dim_high_level_action + j] += weights(i) * n.values()[j] / weight_sum;
      }
    }

    // add total return
    avg_return += trajectory[idx].total_return * weights(i) / weight_sum;
  }

  // normalize
  // mju_scl(parameters_scratch.data(), parameters_scratch.data(), 1.0 / n_elite,
  //         num_parameters);
  // avg_return /= n_elite;

  // loop over elites to compute variance
  std::fill(variance.begin(), variance.end(), 0.0);  // reset variance to zero
  
  for (int t = 0; t < num_spline_points; t++) {
    TimeSpline::Node n = candidate_policy[trajectory_order[0]].plan.NodeAt(t);

    
    for (int j = 0; j < policy.dim_high_level_action; j++) {
      // average
      double p_avg = parameters_scratch[t * policy.dim_high_level_action + j];
      for (int i = 0; i < n_elite; i++) {
        // candidate parameter
        double pi = n.values()[j];
        double diff = pi - p_avg;
        double norm_diff = diff;
        // if (j<23) {
          norm_diff = diff / (model->jnt_range[policy.includeList[j] * 2 + 1] - model->jnt_range[policy.includeList[j] * 2]);
        // }
        
        // variance[t * policy.dim_high_level_action + j] += diff * diff / (n_elite - 1);
        // variance[t * policy.dim_high_level_action + j] += weights(i) * diff * diff / weight_sum;
        variance[t * policy.dim_high_level_action + j] += weights(i) * norm_diff * norm_diff / weight_sum;
        variance[t * policy.dim_high_level_action + j] = std::max(variance[t * policy.dim_high_level_action + j], std_min_ * std_min_);
      }
    }
  }

  

  // std::cout<<"variance mean "<<std::accumulate(variance.begin(), variance.begin()+policy.dim_high_level_action*num_spline_points, 0.0)/policy.dim_high_level_action*num_spline_points<<std::endl;

  // update
  {
    const std::shared_lock<std::shared_mutex> lock(mtx_);
    policy.plan.Clear();
    policy.plan.SetInterpolation(interpolation_);
    for (int t = 0; t < num_spline_points; t++) {
      // absl::Span<const double> values =
      //     absl::MakeConstSpan(parameters_scratch.data() + t * model->nu,
      //                         parameters_scratch.data() + (t + 1) * model->nu);
      absl::Span<const double> values =
          absl::MakeConstSpan(parameters_scratch.data() + t * policy.dim_high_level_action,
                              parameters_scratch.data() + (t + 1) * policy.dim_high_level_action);
      policy.plan.AddNode(times_scratch[t], values);
    }

    // for (int i=0; i < policy.num_spline_points; i++) {
    //   TimeSpline::Node node = policy.plan.NodeAt(i);
    //   // std::cout<<"hip pos "<<i<<" "<<node.values().data()[6]<<" "<<node.values().data()[21]<<std::endl;
    // }
  }

  // improvement: compare nominal to elite average
  improvement =
      mju_max(avg_return - trajectory[trajectory_order[0]].total_return, 0.0);

  // stop timer
  policy_update_compute_time = GetDuration(policy_update_start);

  //store state and policy
  // // double state_copy[85];
  // std::vector<double> state_copy(85);
  // // mju_copy(state_copy, state.data(), 85);
  // std::copy(state.data(), state.data()+85, state_copy.begin());
  // state_buffer.push_back(state_copy);

  // std::vector<double> mirror_state_copy(85);
  // mju_copy(mirror_state_copy.data(), state.data(), 6);
  // mju_copy(mirror_state_copy.data()+6, state.data()+21, 15);
  // mju_copy(mirror_state_copy.data()+21, state.data()+6, 15);
  // mju_copy(mirror_state_copy.data()+36, state.data()+36, 9);
  // mju_scl(mirror_state_copy.data()+37, mirror_state_copy.data()+37, -1, 2);
  // mju_scl(mirror_state_copy.data()+40, mirror_state_copy.data()+40, -1, 2);
  // mju_scl(mirror_state_copy.data()+43, mirror_state_copy.data()+43, -1, 2);
  // mju_copy(mirror_state_copy.data()+45, state.data()+65, 20);
  // mju_copy(mirror_state_copy.data()+65, state.data()+45, 20);
  // mirror_state_buffer.push_back(mirror_state_copy);

  // // double policy_copy[18944];
  // std::vector<double> policy_copy(policy.dim_high_level_action * kMaxTrajectoryHorizon);
  // std::vector<double> mirror_policy_copy(policy.dim_high_level_action * kMaxTrajectoryHorizon);
  // for (int i = 0; i < policy.num_spline_points; i++) {
  //   TimeSpline::Node node = policy.plan.NodeAt(i);
  //   // mju_copy(policy_copy + i * policy.dim_high_level_action, node.values().data(), policy.dim_high_level_action);
  //   std::copy(node.values().data(), node.values().data()+policy.dim_high_level_action, policy_copy.begin()+i*policy.dim_high_level_action);

  //   std::copy(node.values().data()+7, node.values().data()+14, mirror_policy_copy.begin());
  //   std::copy(node.values().data()+7, node.values().data()+7, mirror_policy_copy.begin()+7);
  //   std::copy(node.values().data()+14, node.values().data()+23, mirror_policy_copy.begin()+14);
  //   mju_scl(node.values().data()+15, mirror_state_copy.data()+15, -1, 2);
  //   mju_scl(node.values().data()+18, mirror_state_copy.data()+18, -1, 2);
  //   mju_scl(node.values().data()+21, mirror_state_copy.data()+21, -1, 2);
  //   std::copy(node.values().data()+30, node.values().data()+37, mirror_policy_copy.begin()+23);
  //   std::copy(node.values().data()+23, node.values().data()+30, mirror_policy_copy.begin()+30);
  // }
  // policy_buffer.push_back(policy_copy);
  // mirror_policy_buffer.push_back(mirror_policy_copy);

  // cost_buffer.push_back(avg_return);
  // if (state_buffer.size() > buffer_size) {
  //   state_buffer.pop_front();
  //   policy_buffer.pop_front();
  //   mirror_state_buffer.pop_front();
  //   mirror_policy_buffer.pop_front();
  //   cost_buffer.pop_front();
  // }
  
  // if (state_buffer.size() == buffer_size) {
  //   if (std::accumulate(cost_buffer.begin(), cost_buffer.end(), 0.0) < std::accumulate(best_cost_buffer.begin(), best_cost_buffer.end(), 0.0)) {
  //     std::cout<<state_buffer.size()<<" "<<best_state_buffer.size()<<std::endl;
  //     std::copy(state_buffer.begin(), state_buffer.end(), best_state_buffer.begin());
  //     std::copy(policy_buffer.begin(), policy_buffer.end(), best_policy_buffer.begin());
  //     std::copy(mirror_state_buffer.begin(), mirror_state_buffer.end(), best_state_buffer.begin()+buffer_size); 
  //     std::copy(mirror_policy_buffer.begin(), mirror_policy_buffer.end(), best_policy_buffer.begin()+buffer_size);
  //     std::copy(cost_buffer.begin(), cost_buffer.end(), best_cost_buffer.begin());
  // }
  // std::cout<<"best buffer cost "<<cost_buffer.size()<<" "<<
  //   std::accumulate(best_cost_buffer.begin(), best_cost_buffer.end(), 0.0)<<" "<<
  //   std::accumulate(cost_buffer.begin(), cost_buffer.end(), 0.0)<<std::endl;
  // }
  
  
  // compare cost to the best cost
  
  
  // std::cout<<"state buffer size "<<state_buffer.size()<<" "<<state_buffer[0].size()<<state_buffer[0].data()[0]<<std::endl;
  // std::cout<<"policy buffer size "<<policy_buffer.size()<<" "<<policy_buffer[0].size()<<policy_buffer[0].data()[0]<<std::endl;


}

// compute trajectory using nominal policy
void HierarchicalMPPIPlanner::NominalTrajectory(int horizon) {
  // set policy
  auto nominal_policy = [this, &cp = resampled_policy](
                            double* action, const double* state, double time) {
    // cp.Action(action, state, time);
    SyncPolicyState(cp, state, time);
    cp.HierarchicalAction(action, data_[0].get());
  };

  // rollout nominal policy
  nominal_trajectory.Rollout(nominal_policy, task, model,
                             data_[ThreadPool::WorkerId()].get(), state.data(),
                             time, mocap.data(), userdata.data(), horizon, perturb_std);
}
void HierarchicalMPPIPlanner::NominalTrajectory(int horizon, ThreadPool& pool) {
  NominalTrajectory(horizon);
}

// set action from policy
void HierarchicalMPPIPlanner::ActionFromPolicy(double* action, const double* state,
                                           double time, bool use_previous) {
  const std::shared_lock<std::shared_mutex> lock(mtx_);
  if (use_previous) {
    SyncPolicyState(previous_policy, state, time);
    previous_policy.HierarchicalAction(action, data_[0].get());
    // previous_policy.Action(action, state, time);
  } else {
    SyncPolicyState(policy, state, time);
    policy.HierarchicalAction(action, data_[0].get());
    // policy.Action(action, state, time);
  }
}

void HierarchicalMPPIPlanner::SyncTargetPos(mjData* data) {
  const std::shared_lock<std::shared_mutex> lock(mtx_);
  double* act_winner = new double[model->nq];
  
  // mju_copy(act_winner, data->qpos, 85);
  mju_zero(act_winner, model->nq);
  
  
  double* act_high = new double[policy.dim_high_level_action];
  mju_zero(act_high, policy.dim_high_level_action);
  policy.Action(act_high, nullptr, data->time);
  

  for (int i = 0; i < model->nq; i++) {
    if (i==2) {
      act_winner[i] = data->qpos[model->nq+i];
    }
    else {
      // act_winner[i] = candidate_policy[winner].plan.NodeAt(0).values()[i];
      act_winner[i] = 0;
    }
  }
  for (int i = 0; i < policy.includeList.size(); ++i) {
    act_winner[policy.includeList[i]] = act_high[i];
    // act_winner[policy.includeList[i]] = candidate_policy[winner].plan.NodeAt(0).values()[i];
  }
  for (int i = 0; i < policy.excludeList.size(); ++i) {
    // if (i != 2) {
      act_winner[policy.excludeList[i]] = data->qpos[policy.excludeList[i]];
    // }
  }

  Clamp(act_winner, model->jnt_range, model->nq);

  

    
  // mju_copy(data->qpos+model->nq, act_winner, model->nq);
  // mju_zero(data->qvel+model->nq, model->nq);
  // std::cout<<"copy pos "<<data->qpos[85+6]<<std::endl;
}

// update policy via resampling
void HierarchicalMPPIPlanner::ResamplePolicy(int horizon) {
  // dimensions
  int num_spline_points = resampled_policy.num_spline_points;

  // time
  double nominal_time = time;
  double time_shift = mju_max(
      (horizon - 1) * model->opt.timestep / (num_spline_points - 1), 1.0e-5);
  // printf("time_shift %f\n", time_shift);
  // get spline points
  // std::cout<<"get spline points"<<std::endl;
  for (int t = 0; t < num_spline_points; t++) {
    times_scratch[t] = nominal_time;
    // process nu data at a time
    // resampled_policy.Action(DataAt(parameters_scratch, t * model->nu), nullptr,
    //                         nominal_time);
    // std::cout<<"before resample t: "<<t<<" "<<parameters_scratch.size()<<" "<<t * policy.dim_high_level_action<<std::endl;
    resampled_policy.Action(DataAt(parameters_scratch, t * policy.dim_high_level_action), nullptr,
                            nominal_time);
    // std::cout<<"after resample t: "<<t<<std::endl;
    nominal_time += time_shift;
  }

  // copy resampled policy parameters
  // std::cout<<"copy resample"<<std::endl;
  resampled_policy.plan.Clear();
  for (int t = 0; t < num_spline_points; t++) {
    // absl::Span<const double> values =
    //     absl::MakeConstSpan(parameters_scratch.data() + t * model->nu,
    //                         parameters_scratch.data() + (t + 1) * model->nu);
    absl::Span<const double> values =
        absl::MakeConstSpan(parameters_scratch.data() + t * policy.dim_high_level_action,
                            parameters_scratch.data() + (t + 1) * policy.dim_high_level_action);
    resampled_policy.plan.AddNode(times_scratch[t], values);
  }
  resampled_policy.plan.SetInterpolation(policy.plan.Interpolation());
}

// add random noise to nominal policy
void HierarchicalMPPIPlanner::AddNoiseToPolicy(int i, double std_min) {
  // start timer
  auto noise_start = std::chrono::steady_clock::now();

  // dimensions
  int num_spline_points = candidate_policy[i].num_spline_points;
  // int num_parameters = num_spline_points * model->nu;
  int num_parameters = num_spline_points * policy.dim_high_level_action;

  // sampling token
  absl::BitGen gen_;

  // shift index
  // int shift = i * (model->nu * kMaxTrajectoryHorizon);
  int shift = i * (policy.dim_high_level_action * kMaxTrajectoryHorizon);

  // sample noise
  // variance[k] is the standard deviation for the k^th control parameter over
  // the elite samples we draw a bunch of control actions from this distribution
  // (which i indexes) - the noise is stored in `noise`.
  for (int k = 0; k < num_parameters; k++) {
    noise[k + shift] = absl::Gaussian<double>(
        gen_, 0.0, std::max(std::sqrt(variance[k]), std_min));
  }

  for (int k = 0; k < candidate_policy[i].plan.Size(); k++) {
    TimeSpline::Node n = candidate_policy[i].plan.NodeAt(k);
    // add noise
    // mju_addTo(n.values().data(), DataAt(noise, shift + k * model->nu),
    //           model->nu);
    // // clamp parameters
    // Clamp(n.values().data(), model->actuator_ctrlrange, model->nu);
    // if (i == 3) {
    //   std::cout<<"before add "<<n.values()[0]<<" "<<n.values()[1]<<std::endl;
    // }
    
    
    //mutate based on current noisy policy
    if (i > 1 && k > 0) {
      TimeSpline::Node last_n = candidate_policy[i].plan.NodeAt(k-1);
      mju_copy(n.values().data(), last_n.values().data(), policy.dim_high_level_action);
    }

    // if (i != 0) {
      double* scaled_noise = new double[policy.dim_high_level_action];
      mju_copy(scaled_noise, noise.data()+shift+k*policy.dim_high_level_action, policy.dim_high_level_action);
      for (int j = 0; j < policy.dim_high_level_action; j++) {
        double joint_range = model->jnt_range[2 * policy.includeList[j] + 1] - model->jnt_range[2 * policy.includeList[j]];
        scaled_noise[j] *= joint_range;
      }
      // mju_addTo(n.values().data(), DataAt(noise, shift + k * policy.dim_high_level_action),
      //               policy.dim_high_level_action);
      mju_addTo(n.values().data(), scaled_noise,
                    policy.dim_high_level_action);
    // }

    //mutate based on current noisy policy
    // if (i > 1 && k > 0) {
    //   TimeSpline::Node last_n = candidate_policy[i].plan.NodeAt(k-1);
    //   // mju_copy(n.values().data()+14, last_n.values().data()+14, policy.dim_high_level_action - 14);
    //   mju_copy(n.values().data(), last_n.values().data(), policy.dim_high_level_action);

    //   // mju_copy(n.values().data()+14, last_n.values().data()+14, 9);
    //   // for (int j = 0; j < policy.dim_high_level_action; j++) {
    //   //   if (j < 4 || (j > 6 && j < 11)) {
    //   //     continue;
    //   //   }
    //   //   n.values().data()[j] = last_n.values().data()[j];
    //   // }
    // }
    
    
    // if (i == 3) {
    //   std::cout<<"after add "<<n.values()[0]<<" "<<n.values()[1]<<std::endl;
    // }
    // clamp parameters
    // Clamp(n.values().data(), model->jnt_range, policy.dim_high_level_action);
    for (int j = 0; j < policy.dim_high_level_action; j++) {
      if (n.values().data()[j] > model->jnt_range[2 * policy.includeList[j] + 1]) {
        n.values().data()[j] = model->jnt_range[2 * policy.includeList[j] + 1];
      }
      else if (n.values().data()[j] < model->jnt_range[2 * policy.includeList[j]]) {
        n.values().data()[j] = model->jnt_range[2 * policy.includeList[j]];
      } 
    }

    // if (i == 3) {
    //   std::cout<<"after clamp "<<n.values()[0]<<" "<<n.values()[1]<<std::endl;
    // }

  }

  // end timer
  IncrementAtomic(noise_compute_time, GetDuration(noise_start));
}


//get reference policy
void HierarchicalMPPIPlanner::GetRefPolicy(int i) {
  const std::shared_lock<std::shared_mutex> lock(mtx_);
  // candidate_policy[i].CopyFrom(resampled_policy, resampled_policy.num_spline_points);
  // candidate_policy[i].plan.SetInterpolation(resampled_policy.plan.Interpolation());
  double* current_qpos = new double[model->nq];
  mju_copy(current_qpos, state.data(), model->nq);
  // Eigen::VectorXd current_qpos_eigen = Eigen::Map<Eigen::VectorXd>(current_qpos, 79);
  // std::cout<<"current qpos "<<current_qpos_eigen<<std::endl;
  // int ref_traj_len = 116;
  int min_idx = -1;
  double min_diff = 1e6;
  double* ref_qpos = new double[model->nq]; // exclude xyz

  // int review_size = buffer_size;

  for (int j = 0; j < best_state_buffer.size(); j++) {
    mju_copy(ref_qpos, best_state_buffer[j].data(), model->nq); // only compare joint pos
    Eigen::VectorXd ref_qpos_eigen = Eigen::Map<Eigen::VectorXd>(ref_qpos, model->nq-3);
    // std::cout<<"ref qpos "<<j<<std::endl;
    // std::cout<<"ref qpos "<<ref_qpos_eigen<<std::endl;
    double* ref_diff = new double[policy.dim_high_level_action];
    mju_zero(ref_diff, policy.dim_high_level_action);
    for (int k = 0; k < policy.dim_high_level_action; k++) {
      ref_diff[k] = ref_qpos[policy.includeList[k]] - current_qpos[policy.includeList[k]];
    }
    // mju_sub(ref_diff, current_qpos, ref_qpos, 82);
    double ref_diff_norm = mju_norm(ref_diff, policy.dim_high_level_action);

    if (ref_diff_norm <= min_diff) {
      min_diff = ref_diff_norm;
      min_idx = j;
    }
    
    // std::cout<<"ref diff "<<j<<" "<<ref_diff_norm<<std::endl;
  }
  if (i == 2) {
    min_idx += 1;
    min_idx = mju_min(min_idx, buffer_size-1);
  }
  std::cout<<"min from ref "<<i<<" "<<min_idx<<" "<<min_diff<<std::endl;
  // min_idx = 23;
  // min_idx += 1;
  // double* target_high_act = new double[policy.dim_high_level_action];
  // mju_copy(ref_qpos, model->key_qpos+min_idx*model->nq, 85);
  // for (int j=0; j < policy.dim_high_level_action; j++) {
  //   target_high_act[j] = policy_buffer[min_idx][j];
    
  // }
  // assign to candidate policy
  
  for (int k = 0; k < candidate_policy[i].plan.Size(); k++) {
    TimeSpline::Node n = candidate_policy[i].plan.NodeAt(k);
    mju_copy(n.values().data(), best_policy_buffer[min_idx].data()+k*policy.dim_high_level_action, policy.dim_high_level_action);
    // mju_copy(n.values().data(), target_high_act, policy.dim_high_level_action);
    // std::cout<<"policy "<<n.values().data()[0];
  }
}

void HierarchicalMPPIPlanner::SyncCurrentPos(int i) {
  // double current_qpos[policy.dim_high_level_action];
  std::vector<double> current_qpos;
  current_qpos.resize(policy.dim_high_level_action);
  for (int i = 0; i < policy.dim_high_level_action; i++) {
    current_qpos[i] = state.data()[policy.includeList[i]];
  }
  for (int k = 0; k < candidate_policy[i].plan.Size(); k++) {
    TimeSpline::Node n = candidate_policy[i].plan.NodeAt(k);
    mju_copy(n.values().data(), current_qpos.data(), policy.dim_high_level_action);
  }
}

// compute candidate trajectories
void HierarchicalMPPIPlanner::Rollouts(int num_trajectory, int horizon,
                                   ThreadPool& pool) {
  // reset noise compute time
  noise_compute_time = 0.0;

  // lock std_min
  double std_min = std_min_;

  // random search
  int count_before = pool.GetCount();
  for (int i = 0; i < num_trajectory; i++) {
    pool.Schedule([&s = *this, &model = this->model, &task = this->task,
                   &state = this->state, &time = this->time,
                   &mocap = this->mocap, &userdata = this->userdata, horizon,
                   std_min, i]() {

      double total_ret = 1.0e6;
      while(total_ret >= 1.0e6) {
        // copy nominal policy and sample noise
        {
          const std::shared_lock<std::shared_mutex> lock(s.mtx_);
          s.candidate_policy[i].CopyFrom(s.resampled_policy,
                                        s.resampled_policy.num_spline_points);
          s.candidate_policy[i].plan.SetInterpolation(
              s.resampled_policy.plan.Interpolation());

          // sample noise
          // if (time == 0) {
          //   s.SyncCurrentPos(i);
          // }
          // else {
            if (i == 1) {
            if (s.state_buffer.size() >= s.buffer_size) {
              s.GetRefPolicy(i);
              
            }
            else {
              // s.SyncCurrentPos(i);
              s.AddNoiseToPolicy(i, std_min);
            }
          }
          if (i > 1 && i <= s.current_rollout_num+1) {
            s.SyncCurrentPos(i);
            
          }
          else {
          s.AddNoiseToPolicy(i, std_min);
          }
          // }
          
        }

        // ----- rollout sample policy ----- //

        // policy
        auto sample_policy_i = [&s, &candidate_policy = s.candidate_policy, &i](
                                  double* action, const double* state,
                                  double time) {
          s.SyncPolicyState(candidate_policy[i], state, time);
          candidate_policy[i].HierarchicalAction(action, s.data_[ThreadPool::WorkerId()].get());
          // candidate_policy[i].Action(action, state, time);
          
        };

        // policy rollout
        s.trajectory[i].Rollout(
            sample_policy_i, task, model, s.data_[ThreadPool::WorkerId()].get(),
            state.data(), time, mocap.data(), userdata.data(), horizon, s.perturb_std);
        // if (i == 0 || i == 3 || i == 9) {
        //     std::cout<<"action "<<i<<" "<<s.candidate_policy[i].plan.NodeAt(0).values()[0]<<" "<<s.candidate_policy[i].plan.NodeAt(0).values()[1]<<" "<<s.trajectory[i].total_return<<std::endl;
        //   }
        total_ret = s.trajectory[i].total_return;
        if (total_ret >= 1.0e6) {
          // printf("trajectory %d diverges, resample\n", i);
          printf("trajectory %d diverges\n", i);
        }
        break;
      }

        


    });
  }
  // nominal
  pool.Schedule([&s = *this, horizon]() { s.NominalTrajectory(horizon); });

  // wait
  pool.WaitCount(count_before + num_trajectory + 1);
  pool.ResetCount();
}

// returns the **nominal** trajectory (this is the purple trace)
const Trajectory* HierarchicalMPPIPlanner::BestTrajectory() {
  return &nominal_trajectory;
}

//for act from policy
void HierarchicalMPPIPlanner::SyncPolicyState(HierarchicalPDPolicy policy, const double* src, double time) {
  policy.data_copy->time = time;
  mju_copy(policy.data_copy->qpos, src, model->nq);
  mju_copy(policy.data_copy->qvel, src + model->nq, model->nv);
  mju_copy(policy.data_copy->act,  src + model->nq + model->nv,  model->na);
  mju_copy(policy.data_copy->actuator_length, src + model->nq + model->nv + model->na, model->nu);
  mju_copy(policy.data_copy->actuator_velocity, src + model->nq + model->nv + model->na + model->nu, model->nu);
  
  // copy data for computing gain and bias parameter
  // copy simulation state
  policy.data_copy2->time = time;
  mju_copy(policy.data_copy2->qpos, src, model->nq);
  mju_copy(policy.data_copy2->qvel, src + model->nq, model->nv);
  mju_copy(policy.data_copy2->act,  src + model->nq + model->nv,  model->na);
  mju_copy(policy.data_copy2->actuator_length, src + model->nq + model->nv + model->na, model->nu);
  mju_copy(policy.data_copy2->actuator_velocity, src + model->nq + model->nv + model->na + model->nu, model->nu);


}

// visualize planner-specific traces
void HierarchicalMPPIPlanner::Traces(mjvScene* scn) {
  // sample color
  float color[4];
  color[0] = 1.0;
  color[1] = 1.0;
  color[2] = 1.0;
  color[3] = 1.0;

  // width of a sample trace, in pixels
  double width = GetNumberOrDefault(3, model, "agent_sample_width");

  // scratch
  double zero3[3] = {0};
  double zero9[9] = {0};

  // best
  auto best = this->BestTrajectory();

  // sample traces
  int n_elite = n_elite_;
  for (int k = 0; k < n_elite; k++) {
    // plot sample
    for (int i = 0; i < best->horizon - 1; i++) {
      if (scn->ngeom + task->num_trace > scn->maxgeom) break;
      for (int j = 0; j < task->num_trace; j++) {
        // initialize geometry
        mjv_initGeom(&scn->geoms[scn->ngeom], mjGEOM_LINE, zero3, zero3, zero9,
                     color);

        // elite index
        int idx = trajectory_order[k];
        // make geometry
        mjv_makeConnector(
            &scn->geoms[scn->ngeom], mjGEOM_LINE, width,
            trajectory[idx].trace[3 * task->num_trace * i + 3 * j],
            trajectory[idx].trace[3 * task->num_trace * i + 1 + 3 * j],
            trajectory[idx].trace[3 * task->num_trace * i + 2 + 3 * j],
            trajectory[idx].trace[3 * task->num_trace * (i + 1) + 3 * j],
            trajectory[idx].trace[3 * task->num_trace * (i + 1) + 1 + 3 * j],
            trajectory[idx].trace[3 * task->num_trace * (i + 1) + 2 + 3 * j]);

        // increment number of geometries
        scn->ngeom += 1;
      }
    }
  }
}

// planner-specific GUI elements
void HierarchicalMPPIPlanner::GUI(mjUI& ui) {
  mjuiDef defCrossEntropy[] = {
      {mjITEM_SLIDERINT, "Rollouts", 2, &num_trajectory_, "0 1"},
      {mjITEM_SELECT, "Spline", 2, &interpolation_,
       "Zero\nLinear\nCubic"},
      {mjITEM_SLIDERINT, "Spline Pts", 2, &policy.num_spline_points, "0 1"},
      {mjITEM_SLIDERNUM, "Init. Std", 2, &std_initial_, "0 1"},
      {mjITEM_SLIDERNUM, "Min. Std", 2, &std_min_, "0.01 0.5"},
      {mjITEM_SLIDERINT, "Elite", 2, &n_elite_, "2 128"},
      {mjITEM_END}};

  // set number of trajectory slider limits
  mju::sprintf_arr(defCrossEntropy[0].other, "%i %i", 1, kMaxTrajectory);

  // set spline point limits
  mju::sprintf_arr(defCrossEntropy[2].other, "%i %i", MinSamplingSplinePoints,
                   MaxSamplingSplinePoints);

  // set noise standard deviation limits
  mju::sprintf_arr(defCrossEntropy[3].other, "%f %f", MinNoiseStdDev,
                   MaxNoiseStdDev);

  // add cross entropy planner
  mjui_add(&ui, defCrossEntropy);
}

// planner-specific plots
void HierarchicalMPPIPlanner::Plots(mjvFigure* fig_planner, mjvFigure* fig_timer,
                                int planner_shift, int timer_shift,
                                int planning, int* shift) {
  // ----- planner ----- //
  double planner_bounds[2] = {-6.0, 6.0};

  // improvement
  mjpc::PlotUpdateData(fig_planner, planner_bounds,
                       fig_planner->linedata[0 + planner_shift][0] + 1,
                       mju_log10(mju_max(improvement, 1.0e-6)), 100,
                       0 + planner_shift, 0, 1, -100);

  // legend
  mju::strcpy_arr(fig_planner->linename[0 + planner_shift], "Avg - Best");

  fig_planner->range[1][0] = planner_bounds[0];
  fig_planner->range[1][1] = planner_bounds[1];

  // bounds
  double timer_bounds[2] = {0.0, 1.0};

  // ----- timer ----- //

  PlotUpdateData(
      fig_timer, timer_bounds, fig_timer->linedata[0 + timer_shift][0] + 1,
      1.0e-3 * noise_compute_time * planning, 100, 0 + timer_shift, 0, 1, -100);

  PlotUpdateData(fig_timer, timer_bounds,
                 fig_timer->linedata[1 + timer_shift][0] + 1,
                 1.0e-3 * rollouts_compute_time * planning, 100,
                 1 + timer_shift, 0, 1, -100);

  PlotUpdateData(fig_timer, timer_bounds,
                 fig_timer->linedata[2 + timer_shift][0] + 1,
                 1.0e-3 * policy_update_compute_time * planning, 100,
                 2 + timer_shift, 0, 1, -100);

  // legend
  mju::strcpy_arr(fig_timer->linename[0 + timer_shift], "Noise");
  mju::strcpy_arr(fig_timer->linename[1 + timer_shift], "Rollout");
  mju::strcpy_arr(fig_timer->linename[2 + timer_shift], "Policy Update");

  // planner shift
  shift[0] += 1;

  // timer shift
  shift[1] += 3;
}

}  // namespace mjpc
