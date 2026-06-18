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

#include "mjpc/planners/hierarchical_pd/planner.h"
#include "mjpc/planners/sampling/planner.h"

#include <algorithm>
#include <chrono>
#include <shared_mutex>

#include <absl/random/random.h>
#include <mujoco/mujoco.h>
#include "mjpc/array_safety.h"
#include "mjpc/planners/planner.h"
#include "mjpc/planners/hierarchical_sampling/policy.h"
#include "mjpc/spline/spline.h"
#include "mjpc/states/state.h"
#include "mjpc/task.h"
#include "mjpc/threadpool.h"
#include "mjpc/trajectory.h"
#include "mjpc/utilities.h"

#include <Eigen/Dense>

namespace mjpc {

namespace mju = ::mujoco::util_mjpc;
using mjpc::spline::SplineInterpolation;
using mjpc::spline::TimeSpline;

// initialize data and settings
void HierarchicalPDPlanner::Initialize(mjModel* model, const Task& task) {
  // delete mjData instances since model might have changed.
  data_.clear();
  // allocate one mjData for nominal.
  ResizeMjData(model, 1);

  // model
  this->model = model;

  // task
  this->task = &task;

  // sampling noise std
  noise_exploration[0] = GetNumberOrDefault(0.0, model, "sampling_exploration");

  // optional second std (defaults to 0)
  int se_id = mj_name2id(model, mjOBJ_NUMERIC, "sampling_exploration");
  if (se_id >= 0 && model->numeric_size[se_id] > 1) {
    int se_adr = model->numeric_adr[se_id];
    noise_exploration[1] = model->numeric_data[se_adr+1];
  }

  // set number of trajectories to rollout
  num_trajectory_ = GetNumberOrDefault(50, model, "sampling_trajectories");

  interpolation_ = GetNumberOrDefault(SplineInterpolation::kZeroSpline, model,
                                      "sampling_representation");
  sliding_plan_ = GetNumberOrDefault(0, model, "sampling_sliding_plan");

  if (num_trajectory_ > kMaxTrajectory) {
    mju_error_i("Too many trajectories, %d is the maximum allowed.",
                kMaxTrajectory);
  }

  winner = 0;
}

// allocate memory
void HierarchicalPDPlanner::Allocate() {
  // initial state
  int num_state = model->nq + model->nv + model->na + 2*model->nu + 6*model->nbody;

  // state
  state.resize(num_state);
  mocap.resize(7 * model->nmocap);
  userdata.resize(model->nuserdata);

  // policy
  policy.Allocate(model, *task, kMaxTrajectoryHorizon);
  previous_policy.Allocate(model, *task, kMaxTrajectoryHorizon);
  // plan_scratch = TimeSpline(/*dim=*/model->nu);
  plan_scratch = TimeSpline(/*dim=*/policy.dim_high_level_action);

  // noise
  // noise.resize(kMaxTrajectory * (model->nu * kMaxTrajectoryHorizon));
  noise.resize(kMaxTrajectory * (model->nu * kMaxTrajectoryHorizon));
  // trajectory and parameters
  winner = -1;
  for (int i = 0; i < kMaxTrajectory; i++) {
    trajectory[i].Initialize(num_state, model->nu, task->num_residual,
                             task->num_trace, kMaxTrajectoryHorizon);
    trajectory[i].Allocate(kMaxTrajectoryHorizon);
    candidate_policy[i].Allocate(model, *task, kMaxTrajectoryHorizon);
  }
}

// reset memory to zeros
void HierarchicalPDPlanner::Reset(int horizon,
                            const double* initial_repeated_action) {
  // state
  std::fill(state.begin(), state.end(), 0.0);
  std::fill(mocap.begin(), mocap.end(), 0.0);
  std::fill(userdata.begin(), userdata.end(), 0.0);
  time = 0.0;

  // policy parameters
  policy.Reset(horizon, initial_repeated_action);
  previous_policy.Reset(horizon, initial_repeated_action);

  // scratch
  plan_scratch.Clear();

  // noise
  std::fill(noise.begin(), noise.end(), 0.0);

  // trajectory samples
  for (int i = 0; i < kMaxTrajectory; i++) {
    trajectory[i].Reset(kMaxTrajectoryHorizon);
    candidate_policy[i].Reset(horizon, initial_repeated_action);
  }

  for (const auto& d : data_) {
    if (initial_repeated_action) {
      mju_copy(d->ctrl, initial_repeated_action, model->nu);
    } else {
      mju_zero(d->ctrl, model->nu);
    }
  }

  // improvement
  improvement = 0.0;

  // winner
  winner = 0;
}

// set state
void HierarchicalPDPlanner::SetState(const State& state) {
  state.CopyTo(this->state.data(), this->mocap.data(), this->userdata.data(),
               &this->time);
}

int HierarchicalPDPlanner::OptimizePolicyCandidates(int ncandidates, int horizon,
                                              ThreadPool& pool) {
  // if num_trajectory_ has changed, use it in this new iteration.
  // num_trajectory_ might change while this function runs. Keep it constant
  // for the duration of this function.
  int num_trajectory = num_trajectory_;
  ncandidates = std::min(ncandidates, num_trajectory);
  ResizeMjData(model, pool.NumThreads());

  // ----- rollout noisy policies ----- //
  // start timer
  auto rollouts_start = std::chrono::steady_clock::now();

  // simulate noisy policies
  policy.plan.SetInterpolation(interpolation_);

  // std::cout<<"before Rollouts"<<std::endl;

  this->Rollouts(num_trajectory, horizon, pool);
  // std::cout<<"after Rollouts"<<std::endl;
  // sort candidate policies and trajectories by score
  trajectory_order.clear();
  trajectory_order.reserve(num_trajectory);
  for (int i = 0; i < num_trajectory; i++) {
    trajectory_order.push_back(i);
  }

  // sort so that the first ncandidates elements are the best candidates, and
  // the rest are in an unspecified order
  std::partial_sort(
      trajectory_order.begin(), trajectory_order.begin() + ncandidates,
      trajectory_order.end(), [trajectory = trajectory](int a, int b) {
        return trajectory[a].total_return < trajectory[b].total_return;
      });
  
  // stop timer
  rollouts_compute_time = GetDuration(rollouts_start);

  return ncandidates;
}


//Sync current mj data for policy
void HierarchicalPDPlanner::SyncPolicyData(HierarchicalPDPolicy policy, mjData* src) {

  
  // copy data for inverse dynamic
  // copy simulation state
  // std::cout<<"start copy"<<std::endl;
  policy.data_copy->time = src->time;

  mju_copy(policy.data_copy->qpos, src->qpos, model->nq);
  mju_copy(policy.data_copy->qvel, src->qvel, model->nv);
  mju_copy(policy.data_copy->act,  src->act,  model->na);
  

  // // copy mocap body pose and userdata
  // mju_copy(policy.data_copy->mocap_pos,  src->mocap_pos,  3*model->nmocap);
  // mju_copy(policy.data_copy->mocap_quat, src->mocap_quat, 4*model->nmocap);
  // mju_copy(policy.data_copy->userdata,   src->userdata,   model->nuserdata);

  // copy warm-start acceleration
  mju_copy(policy.data_copy->qacc_warmstart, src->qacc_warmstart, model->nv);
  // std::cout<<"finish copy 1"<<std::endl;

  // copy data for computing gain and bias parameter
  // copy simulation state
  policy.data_copy2->time = src->time;
  mju_copy(policy.data_copy2->qpos, src->qpos, model->nq);
  mju_copy(policy.data_copy2->qvel, src->qvel, model->nv);
  mju_copy(policy.data_copy2->act,  src->act,  model->na);
  mju_copy(policy.data_copy2->ctrl,  src->ctrl,  model->nu);

  // copy warm-start acceleration
  mju_copy(policy.data_copy2->qacc_warmstart, src->qacc_warmstart, model->nv);


}


//for act from policy
void HierarchicalPDPlanner::SyncPolicyState(HierarchicalPDPolicy policy, const double* src, double time) {

  
  // copy data for inverse dynamic
  // copy simulation state
  // std::cout<<"start copy"<<std::endl;
  policy.data_copy->time = time;

  mju_copy(policy.data_copy->qpos, src, model->nq);
  mju_copy(policy.data_copy->qvel, src + model->nq, model->nv);
  mju_copy(policy.data_copy->act,  src + model->nq + model->nv,  model->na);
  mju_copy(policy.data_copy->actuator_length, src + model->nq + model->nv + model->na, model->nu);
  mju_copy(policy.data_copy->actuator_velocity, src + model->nq + model->nv + model->na + model->nu, model->nu);
  

  // mj_forward(model, policy.data_copy);
  // // copy mocap body pose and userdata
  // mju_copy(policy.data_copy->mocap_pos,  src->mocap_pos,  3*model->nmocap);
  // mju_copy(policy.data_copy->mocap_quat, src->mocap_quat, 4*model->nmocap);
  // mju_copy(policy.data_copy->userdata,   src->userdata,   model->nuserdata);

  // copy warm-start acceleration
  // mju_copy(policy.data_copy->qacc_warmstart, src->qacc_warmstart, model->nv);
  // std::cout<<"finish copy 1"<<std::endl;

  // copy data for computing gain and bias parameter
  // copy simulation state
  policy.data_copy2->time = time;
  mju_copy(policy.data_copy2->qpos, src, model->nq);
  mju_copy(policy.data_copy2->qvel, src + model->nq, model->nv);
  mju_copy(policy.data_copy2->act,  src + model->nq + model->nv,  model->na);
  mju_copy(policy.data_copy2->actuator_length, src + model->nq + model->nv + model->na, model->nu);
  mju_copy(policy.data_copy2->actuator_velocity, src + model->nq + model->nv + model->na + model->nu, model->nu);
  // mj_forward(model, policy.data_copy2);


}


// optimize nominal policy using random sampling
void HierarchicalPDPlanner::OptimizePolicy(int horizon, ThreadPool& pool) {
  // resample nominal policy to current time
  // std::cout << "UpdateNominalPolicy" << std::endl;
  this->UpdateNominalPolicy(horizon);

  // double act_current = policy.plan.NodeAt(0).values()[6];
  // std::cout<<"current policy "<<" "<<act_current<<std::endl;
 
  OptimizePolicyCandidates(1, horizon, pool);
  
  // // ----- update policy ----- //
  // // start timer
  auto policy_update_start = std::chrono::steady_clock::now();
  // std::cout << "CopyCandidateToPolicy" << std::endl;
  // std::cout<<"before copy "<<policy.plan.NodeAt(0).values()[3]<<std::endl;
  CopyCandidateToPolicy(0);
  // std::cout<<"after copy "<<policy.plan.NodeAt(0).values()[3]<<std::endl;
  // for (int i=0; i < policy.num_spline_points; i++) {
  //     TimeSpline::Node node = policy.plan.NodeAt(i);
  //     std::cout<<"hip pos "<<i<<" "<<node.values().data()[6]<<" "<<node.values().data()[21]<<std::endl;
  //   }
  // for (int i = 0; i < num_trajectory_; i++) {
  //   printf("total return of trajectory %d %f\n", i, trajectory[i].total_return);
  //   break;
  // }
  // // improvement: compare nominal to winner
  double best_return = trajectory[0].total_return;
  improvement = mju_max(best_return - trajectory[winner].total_return, 0.0);

  // // stop timer
  policy_update_compute_time = GetDuration(policy_update_start);
  // std::cout << "OptimizePolicy" << std::endl;
}

// compute trajectory using nominal policy
void HierarchicalPDPlanner::NominalTrajectory(int horizon, ThreadPool& pool) {
  // set policy
  auto nominal_policy = [this, &cp = candidate_policy[0]](
                            double* action, const double* state, double time) {
    // cp.Action(action, state, time);
    // time_t start = clock();
    // cp.data_copy = mj_copyData(cp.data_copy, this->model, data_[0].get());
    // cp.data_copy2 = mj_copyData(cp.data_copy2, this->model, data_[0].get());
    // SyncPolicyData(cp, data_[0].get());
    
    SyncPolicyState(cp, state, time);
    // std::cout<<"get copy used "<<(double)(clock() - start)/CLOCKS_PER_SEC<<std::endl;
    // std::cout<<"NomialTrajectory"<<std::endl;
    // std::cout<<"action "<<action[candidate_policy[0].includeList[20]]<<std::endl;
    cp.HierarchicalAction(action, data_[0].get());

  };



  // rollout nominal policy
  trajectory[0].Rollout(nominal_policy, task, model, data_[0].get(),
                        state.data(), time, mocap.data(), userdata.data(),
                        horizon);
}

// set action from policy
void HierarchicalPDPlanner::ActionFromPolicy(double* action, const double* state,
                                       double time, bool use_previous) {
  const std::shared_lock<std::shared_mutex> lock(mtx_);
  
  if (use_previous) {
    // previous_policy.Action(action, state, time);
    // std::cout<<"use previous policy"<<std::endl;
    // previous_policy.data_copy = mj_copyData(previous_policy.data_copy, model, data_[0].get());
    // previous_policy.data_copy2 = mj_copyData(previous_policy.data_copy2, model, data_[0].get());
    std::cout<<"use previous policy"<<std::endl;
    // SyncPolicyData(previous_policy, data_[0].get());
    SyncPolicyState(previous_policy, state, time);
    previous_policy.HierarchicalAction(action, data_[0].get());
  } else {
    // policy.Action(action, state, time);
    // policy.data_copy = mj_copyData(policy.data_copy, model, data_[0].get());
    // policy.data_copy2 = mj_copyData(policy.data_copy2, model, data_[0].get());
    // std::cout<<"use current policy"<<std::endl;
    

    // Eigen::VectorXd qvel = Eigen::Map<Eigen::VectorXd>(data_[0].get()->qvel, model->nq);
    // Eigen::VectorXd qpos = Eigen::Map<Eigen::VectorXd>(data_[0].get()->qpos, model->nq);
    // std::cout<<"pos and vel "<<qvel.maxCoeff()<<" "<<qvel.minCoeff()<<" "<<qpos.maxCoeff()<<" "<<qpos.minCoeff()<<std::endl;


    // Eigen::VectorXd qpos2(model->nq);
    // for (int i = 0; i < model->nq; i++) {
    //   qpos2(i) = state[i];
    // }
    
    // Eigen::VectorXd qvel2(model->nq);
    // for (int i = 0; i < model->nq; i++) {
    //   qpos2(i) = state[i+model->nq];
    // }
    


    // std::cout<<"pos and vel state"<<qvel2.maxCoeff()<<" "<<qvel2.minCoeff()<<" "<<qpos2.maxCoeff()<<" "<<qpos2.minCoeff()<<std::endl;

    // std::cout<<"time "<<time<<std::endl;
    // double act_winner = candidate_policy[winner].plan.NodeAt(0).values()[6];
    
    SyncPolicyState(policy, state, time);
    
    // SyncPolicyData(policy, data_[0].get());
    policy.HierarchicalAction(action, data_[0].get());
    // std::cout<<action[100]<<std::endl;
  }
}

void HierarchicalPDPlanner::SyncTargetPos(mjData* data) {
  const std::shared_lock<std::shared_mutex> lock(mtx_);
  double act_winner[85];
  
  // mju_copy(act_winner, data->qpos, 85);
  mju_zero(act_winner, 85);
  
  
  double* act_high = new double[policy.dim_high_level_action];
  mju_zero(act_high, policy.dim_high_level_action);
  policy.Action(act_high, nullptr, data->time);
  

  for (int i = 0; i < 85; i++) {
    if (i==2) {
      act_winner[i] = data->qpos[85+i];
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

  Clamp(act_winner, model->jnt_range, 85);

  

    
  mju_copy(data->qpos+85, act_winner, 85);
  mju_zero(data->qvel+85, 85);
  // std::cout<<"copy pos "<<data->qpos[85+6]<<std::endl;
}

// update policy via resampling
void HierarchicalPDPlanner::UpdateNominalPolicy(int horizon) {
  // dimensions
  int num_spline_points = candidate_policy[winner].num_spline_points;

  // set time
  double nominal_time = time;
  double time_horizon = (horizon - 1) * model->opt.timestep;

  if (sliding_plan_) {
    // extra points required outside of the horizon window
    int extra_points;
    switch (interpolation_) {
      case spline::SplineInterpolation::kZeroSpline:
        extra_points = 1;
        break;
      case spline::SplineInterpolation::kLinearSpline:
        extra_points = 2;
        break;
      case spline::SplineInterpolation::kCubicSpline:
        extra_points = 4;
        break;
    }

    // temporal distance between spline points
    double time_shift;
    if (num_spline_points > extra_points) {
      time_shift = mju_max(time_horizon /
                            (num_spline_points - extra_points), 1.0e-5);
    } else {
      // not a valid setting, but avoid division by zero
      time_shift = time_horizon;
    }

    const std::shared_lock<std::shared_mutex> lock(mtx_);
    policy.plan.DiscardBefore(nominal_time);
    if (policy.plan.Size() == 0) {
      policy.plan.AddNode(time);
    }
    while (policy.plan.Size() < num_spline_points) {
      // duplicate the last node, with a time further in the future.
      double new_node_time = (policy.plan.end() - 1)->time() + time_shift;
      TimeSpline::Node new_node = policy.plan.AddNode(new_node_time);
      std::copy((policy.plan.end() - 2)->values().begin(),
                (policy.plan.end() - 2)->values().end(),
                new_node.values().begin());
    }
  } else {
    // non-sliding, resample the plan into a scratch plan
    double time_shift;
    if (interpolation_ == spline::SplineInterpolation::kZeroSpline) {
      time_shift = mju_max(time_horizon / num_spline_points, 1.0e-5);
    } else {
      time_shift = mju_max(time_horizon / (num_spline_points - 1), 1.0e-5);
    }

    // resample the nominal plan on a new set of spline points
    plan_scratch.Clear();
    plan_scratch.SetInterpolation(interpolation_);
    plan_scratch.Reserve(num_spline_points);
    // std::cout<<"update nominal policy"<<std::endl;
    // get spline points
    // for (int t = 0; t < num_spline_points; t++) {
    //   double act_current = candidate_policy[winner].plan.NodeAt(t).values()[6];
    //   std::cout<<"winner policy "<<t<<" "<<act_current<<std::endl;
    // }
   
    for (int t = 0; t < num_spline_points; t++) {
      //TODO: double check this part
      TimeSpline::Node node = plan_scratch.AddNode(nominal_time);
      candidate_policy[winner].Action(node.values().data(), /*state=*/nullptr,
                                      nominal_time);
      
      // std::cout<<"action "<<t<<std::endl;
      // std::cout<<"winner "<<winner<<std::endl;
      // candidate_policy[winner].data_copy = mj_copyData(candidate_policy[winner].data_copy, model, data_[0].get());
      // candidate_policy[winner].data_copy2 = mj_copyData(candidate_policy[winner].data_copy2, model, data_[0].get());
      // SyncPolicyData(candidate_policy[winner], data_[winner].get());
      // candidate_policy[winner].HierarchicalAction(node.values().data(), data_[winner].get());
      nominal_time += time_shift;
    }

    // copy scratch into plan
    {
      const std::shared_lock<std::shared_mutex> lock(mtx_);
      policy.plan = plan_scratch;
    }
  }
}

// add random noise to nominal policy
void HierarchicalPDPlanner::AddNoiseToPolicy(double start_time, int i) {
  // start timer
  auto noise_start = std::chrono::steady_clock::now();

  // sampling token
  absl::BitGen gen_;

  // get standard deviation, fixed or mixture of noise_exploration[0,1]
  double std = noise_exploration[0];
  constexpr double kStd2Proportion = 0.2;  // hardcoded proportion of 2nd std
  if (noise_exploration[1] > 0 && absl::Bernoulli(gen_, kStd2Proportion)) {
    std = noise_exploration[1];
  }

  for (const TimeSpline::Node& node : candidate_policy[i].plan) {
    for (int k = 0; k < policy.dim_high_level_action; k++) {
      // double scale = 0.5 * (model->actuator_ctrlrange[2 * k + 1] -
      //                       model->actuator_ctrlrange[2 * k]);
      double scale = 0.5 * (model->jnt_range[2 * policy.includeList[k] + 1] -
                            model->jnt_range[2 * policy.includeList[k]]);
      // for (int k = 0; k < policy.dim_high_level_action; k++) {
      if (scale > 1.57) {
        scale = 1.57;
      }
      // }
      // if (k == 6 || 
      // k == 7 || k == 8 || 
      // // k == 12 || k == 15 || 
      // k == 21 
      // || k == 22 || k == 23 
      // // || k == 27 || k == 30
      // ) {
      //   std = 0.2;
      // }
      double noise = absl::Gaussian<double>(gen_, 0.0, scale * std);
      // if (k == 3) {
      //   std::cout<<"noise "<<k<<" "<<scale<<" "<<noise<<" "<<node.values()[k] <<std::endl;
      // }
      
      node.values()[k] += noise;
      if (node.values()[k] > model->jnt_range[2 * policy.includeList[k] + 1]) {
        node.values()[k] = model->jnt_range[2 * policy.includeList[k] + 1];
      }
      else if (node.values()[k] < model->jnt_range[2 * policy.includeList[k]]) {
        node.values()[k] = model->jnt_range[2 * policy.includeList[k]];
      }
      // if (k == 6) {
      //   std::cout<<"right hip pos"<<node.values()[k]<<" "<<noise<<" "<<scale<<std::endl;
      // }
      // if (k == 21) {
      //   std::cout<<"left hip pos"<<node.values()[k]<<" "<<noise<<" "<<scale<<std::endl;
      // }
    }
    // TODO: clamp using high level bounds
    // Clamp(node.values().data(), model->actuator_ctrlrange, model->nu);
  }

  // end timer
  IncrementAtomic(noise_compute_time, GetDuration(noise_start));
}

// compute candidate trajectories
void HierarchicalPDPlanner::Rollouts(int num_trajectory, int horizon,
                               ThreadPool& pool) {
  // reset noise compute time
  noise_compute_time = 0.0;

  // random search
  int count_before = pool.GetCount();
  for (int i = 0; i < num_trajectory; i++) {
    pool.Schedule([&s = *this, &model = this->model, &task = this->task,
                   &state = this->state, &time = this->time,
                   &mocap = this->mocap, &userdata = this->userdata, horizon,
                   i]() {
      double total_ret = 1.0e6;
      while(total_ret >= 1.0e6) {
        // copy nominal policy
        {
          const std::shared_lock<std::shared_mutex> lock(s.mtx_);
          s.candidate_policy[i].CopyFrom(s.policy, s.policy.num_spline_points);
        }
      
        // sample noise policy
        // std::cout<<"before AddNoiseToPolicy"<<i<<std::endl;
        if (i != 0) s.AddNoiseToPolicy(time, i);
        // double act_current = s.candidate_policy[i].plan.NodeAt(0).values()[6];
        // std::cout<<"candidate policy "<<" "<<act_current<<std::endl;
        // std::cout<<"after AddNoiseToPolicy"<<i<<std::endl;
        // ----- rollout sample policy ----- //

        // policy lambda formulation
        auto sample_policy_i = [&s, &candidate_policy = s.candidate_policy, &i](
                                   double* action, const double* state,
                                   double time) {
          // candidate_policy[i].Action(action, state, time);
          // s.candidate_policy[i].data_copy = mj_copyData(s.candidate_policy[i].data_copy, s.model, s.data_[ThreadPool::WorkerId()].get());
          // s.candidate_policy[i].data_copy2 = mj_copyData(s.candidate_policy[i].data_copy2, s.model, s.data_[ThreadPool::WorkerId()].get());
          // s.SyncPolicyData(candidate_policy[i], s.data_[ThreadPool::WorkerId()].get());
          s.SyncPolicyState(candidate_policy[i], state, time);
          // std::cout<<"Rollouts"<<std::endl;
          candidate_policy[i].HierarchicalAction(action, s.data_[ThreadPool::WorkerId()].get());
        };

        // policy rollout
        s.trajectory[i].Rollout(
            sample_policy_i, task, model, s.data_[ThreadPool::WorkerId()].get(),
            state.data(), time, mocap.data(), userdata.data(), horizon);
        total_ret = s.trajectory[i].total_return;
        if (total_ret >= 1.0e6) {
          printf("trajectory %d diverges, resample\n", i);
        }
        
      }
     
    });
  }
  pool.WaitCount(count_before + num_trajectory);
  pool.ResetCount();
}

// return trajectory with best total return
const Trajectory* HierarchicalPDPlanner::BestTrajectory() {
  return winner >= 0 ? &trajectory[winner] : nullptr;
}

// visualize planner-specific traces
void HierarchicalPDPlanner::Traces(mjvScene* scn) {
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
  for (int k = 0; k < num_trajectory_; k++) {
    // skip winner
    if (k == winner) continue;

    // plot sample
    for (int i = 0; i < best->horizon - 1; i++) {
      if (scn->ngeom + task->num_trace > scn->maxgeom) break;
      for (int j = 0; j < task->num_trace; j++) {
        // initialize geometry
        mjv_initGeom(&scn->geoms[scn->ngeom], mjGEOM_LINE, zero3, zero3, zero9,
                     color);

        // make geometry
        mjv_makeConnector(
            &scn->geoms[scn->ngeom], mjGEOM_LINE, width,
            trajectory[k].trace[3 * task->num_trace * i + 3 * j],
            trajectory[k].trace[3 * task->num_trace * i + 1 + 3 * j],
            trajectory[k].trace[3 * task->num_trace * i + 2 + 3 * j],
            trajectory[k].trace[3 * task->num_trace * (i + 1) + 3 * j],
            trajectory[k].trace[3 * task->num_trace * (i + 1) + 1 + 3 * j],
            trajectory[k].trace[3 * task->num_trace * (i + 1) + 2 + 3 * j]);

        // increment number of geometries
        scn->ngeom += 1;
      }
    }
  }
}

// planner-specific GUI elements
void HierarchicalPDPlanner::GUI(mjUI& ui) {
  mjuiDef defSampling[] = {
      {mjITEM_SLIDERINT, "Rollouts", 2, &num_trajectory_, "0 1"},
      {mjITEM_SELECT, "Spline", 2, &interpolation_,
       "Zero\nLinear\nCubic"},
      {mjITEM_SLIDERINT, "Spline Pts", 2, &policy.num_spline_points, "0 1"},
      {mjITEM_SLIDERNUM, "Noise Std", 2, noise_exploration, "0 1"},
      {mjITEM_SLIDERNUM, "Noise Std2", 2, noise_exploration+1, "0 1"},
      {mjITEM_CHECKBYTE, "Sliding plan", 2, &sliding_plan_, ""},
      {mjITEM_END}};

  // set number of trajectory slider limits
  mju::sprintf_arr(defSampling[0].other, "%i %i", 1, kMaxTrajectory);

  // set spline point limits
  mju::sprintf_arr(defSampling[2].other, "%i %i", MinSamplingSplinePoints,
                   MaxSamplingSplinePoints);

  // set noise standard deviation limits
  mju::sprintf_arr(defSampling[3].other, "%f %f", MinNoiseStdDev,
                   MaxNoiseStdDev);

  // add sampling planner
  mjui_add(&ui, defSampling);
}

// planner-specific plots
void HierarchicalPDPlanner::Plots(mjvFigure* fig_planner, mjvFigure* fig_timer,
                            int planner_shift, int timer_shift, int planning,
                            int* shift) {
  // ----- planner ----- //
  double planner_bounds[2] = {-6.0, 6.0};

  // improvement
  mjpc::PlotUpdateData(fig_planner, planner_bounds,
                       fig_planner->linedata[0 + planner_shift][0] + 1,
                       mju_log10(mju_max(improvement, 1.0e-6)), 100,
                       0 + planner_shift, 0, 1, -100);

  // legend
  mju::strcpy_arr(fig_planner->linename[0 + planner_shift], "Improvement");

  fig_planner->range[1][0] = planner_bounds[0];
  fig_planner->range[1][1] = planner_bounds[1];

  // bounds
  double timer_bounds[2] = {0.0, 1.0};

  // ----- timer ----- //

  PlotUpdateData(fig_timer, timer_bounds,
                 fig_timer->linedata[0 + timer_shift][0] + 1,
                 1.0e-3 * noise_compute_time * planning, 100,
                 0 + timer_shift, 0, 1, -100);

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

double HierarchicalPDPlanner::CandidateScore(int candidate) const {
  return trajectory[trajectory_order[candidate]].total_return;
}

// set action from candidate policy
void HierarchicalPDPlanner::ActionFromCandidatePolicy(double* action, int candidate,
                                                const double* state,
                                                double time) {
  // candidate_policy[trajectory_order[candidate]].Action(action, state, time);
  // candidate_policy[trajectory_order[candidate]].data_copy = mj_copyData(candidate_policy[trajectory_order[candidate]].data_copy, model, data_[0].get());
  // candidate_policy[trajectory_order[candidate]].data_copy2 = mj_copyData(candidate_policy[trajectory_order[candidate]].data_copy2, model, data_[0].get());
  // SyncPolicyData(candidate_policy[trajectory_order[candidate]], data_[0].get());
  SyncPolicyState(candidate_policy[trajectory_order[candidate]], state, time);
  std::cout<<"ActionFromCandidatePolicy"<<std::endl;
  candidate_policy[trajectory_order[candidate]].HierarchicalAction(action, data_[0].get());
}

void HierarchicalPDPlanner::CopyCandidateToPolicy(int candidate) {
  // set winner
  winner = trajectory_order[candidate];

  {
    const std::shared_lock<std::shared_mutex> lock(mtx_);
    previous_policy = policy;
    // std::cout<<"copy dim "<<candidate_policy[winner].plan.Dim()<<" "<<policy.plan.Dim()<<std::endl;
    // policy = candidate_policy[winner];
    
    policy.CopyFrom(candidate_policy[winner], policy.num_spline_points);

    


    // double act_winner = candidate_policy[winner].plan.NodeAt(0).values()[6];
    // double act_current = policy.plan.NodeAt(0).values()[6];
    // std::cout<<"copy "<<act_winner<<" "<<act_current<<std::endl;
  }
}
}  // namespace mjpc
