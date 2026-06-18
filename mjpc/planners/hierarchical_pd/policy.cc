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

#include "mjpc/planners/hierarchical_pd/policy.h"

#include <absl/random/random.h>

#include <absl/log/check.h>
#include <absl/types/span.h>
#include <mujoco/mujoco.h>
#include "mjpc/spline/spline.h"
#include "mjpc/task.h"
#include "mjpc/trajectory.h"
#include "mjpc/utilities.h"
#include <fstream>
#include <OsqpEigen/OsqpEigen.h>
#include <Eigen/Dense>
#include <time.h>
// #include <glpk.h>


namespace mjpc {

using mjpc::spline::TimeSpline;

// allocate memory
void HierarchicalPDPolicy::Allocate(const mjModel* model, const Task& task,
                              int horizon) {
  // model
  this->model = model;

  // spline points
  num_spline_points = GetNumberOrDefault(kMaxTrajectoryHorizon, model,
                                         "sampling_spline_points");

  if (model->nu < 100) { // hand
    excludeList = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 42, 43, 44, 45, 46, 47, 48};
    includeList = { 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41}; 
  } else if (model->nu == 120) { // ostrich
      excludeList = {0, 1, 2, 3, 4, 5, 22, 23, 26, 27, 30, 31, 34, 35, 38, 39, 42, 43, 46, 47, 50, 51, 54, 55};
      includeList = {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 28, 29, 32, 33, 36, 37, 40, 41, 44, 45, 48, 49, 52, 53}; 
  } else { // human
      excludeList = {0, 1, 2, 3, 4, 5, 9, 10, 11, 13, 14, 18, 19, 20, 24, 25, 26, 28, 29, 33, 34, 35, 
  45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 57, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 77, 83, 84};

      includeList = {6, 7, 8, 12, 15, 16, 17, 21, 22, 23, 27, 30, 31, 32, 
  36, 37, 38, 39, 40, 41, 42, 43, 44, 
  55, 56, 58, 59, 60, 61, 62, 75, 76, 78, 79, 80, 81, 82}; 
  }

  dim_high_level_action = includeList.size();  // number of high-level actions, set to joint number for MS model
  high_level_actions.resize(kMaxTrajectoryHorizon * dim_high_level_action);   // (horizon-1 x num_action)

  // plan = TimeSpline(/*dim=*/model->nu);
  plan = TimeSpline(/*dim=*/dim_high_level_action);
  plan.Reserve(num_spline_points);
  
	// initialize data copy
  data_copy = mj_makeData(model);
	data_copy2 = mj_makeData(model); 
}

// reset memory to zeros
void HierarchicalPDPolicy::Reset(int horizon, const double* initial_repeated_action) {
  plan.Clear();
  if (initial_repeated_action != nullptr) {
    plan.AddNode(0, absl::MakeConstSpan(initial_repeated_action, dim_high_level_action));
  }
}

// set action from policy
void HierarchicalPDPolicy::Action(double* action, const double* state,
                            double time) const {
  CHECK(action != nullptr);
  plan.Sample(time, absl::MakeSpan(&action[0], dim_high_level_action));
}

void HierarchicalPDPolicy::HierarchicalAction(double* action, mjData* data) const {
  CHECK(action != nullptr);
  
  //TODO: sample high-level action and then convert to low-level action
  std::vector<double> high_level_action(dim_high_level_action);
  // std::cout<<"before sample "<<high_level_action[3]<<std::endl;
  plan.Sample(data_copy->time, absl::MakeSpan(&high_level_action[0], dim_high_level_action));
  Eigen::VectorXd high_act_vec = Eigen::Map<Eigen::VectorXd>(&high_level_action[0], dim_high_level_action);
  // std::cout<<"after sample "<<high_act_vec.sum()<<std::endl;
  // Eigen::VectorXd qvel = Eigen::Map<Eigen::VectorXd>(data->qvel, model->nu);
  // Eigen::VectorXd qpos = Eigen::Map<Eigen::VectorXd>(data->qpos, model->nu);

  
  // std::cout<<"pos and vel "<<qvel.maxCoeff()<<" "<<qvel.minCoeff()<<" "<<qpos.maxCoeff()<<" "<<qpos.minCoeff()<<std::endl;
  
  HighToLowAction(&high_level_action[0], action, data);
  // Clamp controls
  Eigen::VectorXd act_vec = Eigen::Map<Eigen::VectorXd>(&action[0], model->nu);
  Eigen::VectorXd act_range = Eigen::Map<Eigen::VectorXd>(model->actuator_ctrlrange, model->nu);

  // std::cout<<act_vec.maxCoeff()<<" "<<act_vec.minCoeff()<<" "<<act_range.maxCoeff()<<" "<<act_range.minCoeff()<<std::endl;
  Clamp(action, model->actuator_ctrlrange, model->nu);
}

void HierarchicalPDPolicy::HighToLowAction(double* high_level_action, double* action, mjData* data) const {
  CHECK(action != nullptr);
  // absl::BitGen gen_;

  // Clamp(high_level_action, model->jnt_range, model->nq);
  // Eigen::VectorXd high_act_vec = Eigen::Map<Eigen::VectorXd>(high_level_action, model->nv);
  // std::cout<<"high level action "<<high_act_vec.maxCoeff()<<" "<<high_act_vec.minCoeff()<<std::endl;
  // mju_zero(high_level_action, dim_high_level_action);
  // high_level_action[4] = 1.22;
  // high_level_action[231] = 1.57;
  // high_level_action[15] = -1.5;
  // position
  // high_level_action[6] = -0.3;
  // high_level_action[15] = -0.3;
  // high_level_action[21] = 0.5;
  // high_level_action[27] = 1.57;
  // high_level_action[56] = 3.14;
  // high_level_action[76] = 3.14;
  // if (data->time < 0.5) {
  //   high_level_action[56] = 1.57;
  // }
  // else {
  //   high_level_action[56] = 0;

  // }
  
  // high_level_action[76] = 1.57;
  // if (data->time)
  
  

  // high_level_action[56] = 2;
  // high_level_action[76] = 2;

  // restrict the next position given the current position
  // for (int i=0; i<model->nv; i++) {
  //   // double pos_diff = high_level_action[i] - data->qpos[i];
  //   // if (pos_diff > 3*model->opt.timestep) {
  //   //   high_level_action[i] = data->qpos[i] + 3*model->opt.timestep;
  //   // }
  //   // if (pos_diff < -3*model->opt.timestep) {
  //   //   high_level_action[i] = data->qpos[i] - 3*model->opt.timestep;
  //   // }
  //   // if (i < 6) {
  //     high_level_action[i] = data_copy->qpos[i];
  //   // }
  //   // high_level_action[i] = data_copy->qpos[i];
  // }

  // assign current qpos to excluded joint
  // for (int i = 0; i < excludeList.size(); ++i) {
  //       high_level_action[excludeList[i]] = data_copy->qpos[excludeList[i]];
  // }
  // const int nq = model->nq;
  double* target_pos = new double[model->nq];
  mju_zero(target_pos, model->nq);
  for (int i = 0; i < excludeList.size(); ++i) {
    target_pos[excludeList[i]] = data_copy->qpos[excludeList[i]];
  }
  // mju_copy(target_pos, data_copy->qpos, 85);
  
  for (int i = 0; i < includeList.size(); ++i) {
    target_pos[includeList[i]] = high_level_action[i];
    // std::cout<<"include "<<includeList[i]<<" "<<high_level_action[i]<<std::endl;
  }

  // for (int i = 0; i < spinalList.size(); ++i) {
  //   target_pos[spinalList[i]] = data_copy->qpos[spinalList[i]];
  // }
  // std::cout<<"include "<<includeList[20]<<" "<<high_level_action[includeList[20]]<<std::endl;
  
  // mju_error_i(
  //     "run end here", 0
  //     );

  // if (data_copy->time > 100){
  //   std::cout<<" action "<<high_level_action[6]<<std::endl;
  // }
  // Open a file in write mode
    // std::ofstream outFile("high_level_action.txt");

    // // Check if the file was successfully opened
    // if (!outFile) {
    //     std::cerr << "Error: Could not open the file for writing!" << std::endl;
    //     // return 1;
    // }

    
  

  

  Clamp(target_pos, model->jnt_range, model->nq);
  
  // mju_zero(target_pos, model->nq);
  // target_pos[13] = 1;
  // target_pos[36] = 1;
  // if (data_copy->time == 0) {
  //   Eigen::VectorXd target_qpos_vec = Eigen::Map<Eigen::VectorXd>(target_qpos, 85);
  //   Eigen::VectorXd current_qpos_vec = Eigen::Map<Eigen::VectorXd>(data_copy->qpos, model->nq);
  //   std::cout<<"qpos diff"<<target_qpos_vec.sum()<<" "<<current_qpos_vec.sum()<<std::endl;
  // }
  // for (int i = 0; i < 85; ++i) {
  //   std::cout<<"target pos "<<i<<" "<<target_pos[i]<<std::endl;
  // }
  
  // mju_copy(model->key_qpos, high_level_action, 85);
  // std::cout<<"copy key "<<model->key_qpos[6]<<std::endl;
  // Write each double value to the file, one per line
    // for (int i=0; i<85; i++) {
    //     outFile << high_level_action[i] << std::endl;
    // }

    // Close the file
  
  // outFile.close();
  Eigen::VectorXd action_vec = get_ctrl(target_pos);

  // velocity
  // for (int i=0; i<model->nv; i++) {
  //   high_level_action[i] = -data_copy->qpos[i]/model->opt.timestep;
  // }
  // // high_level_action[6] = 2;
  // high_level_action[56] = (3.14-data_copy->qpos[56])/model->opt.timestep;
  // high_level_action[76] = (3.14-data_copy->qpos[76])/model->opt.timestep;
  // // high_level_action[56] = 3;
  // // high_level_action[56] = 3;
  // for (int i=0; i<model->nv; i++) {
  //   if (high_level_action[i] > 100) {
  //     high_level_action[i] = 100;
  //   }
  //   if (high_level_action[i] < -100) {
  //     high_level_action[i] = -1;
  //   }
  // }

  // Eigen::VectorXd action_vec = get_ctrl2(high_level_action);
  


  
  
  
  // std::cout<<"get ctrl used "<<(double)(clock() - start)/CLOCKS_PER_SEC<<std::endl;

  for (int k = 0; k < model->nu; k++) {
      action[k] =action_vec(k);
    }
  // std::cout<<"action "<<action_vec.sum()<<std::endl;

}

// get control from mj data
Eigen::VectorXd HierarchicalPDPolicy::get_mus_ctrl() const {
  Eigen::VectorXd act0 = Eigen::Map<Eigen::VectorXd>(data_copy->act, model->nu);
  Eigen::VectorXd one_vec = Eigen::VectorXd::Ones(model->nu);

  double ts = model->opt.timestep;
  Eigen::VectorXd tA = 0.01 * (0.5*one_vec.array() + 1.5 * act0.array());
  Eigen::VectorXd tD = 0.04 / (0.5*one_vec.array() + 1.5 * act0.array());
  Eigen::VectorXd tausmooth = 5 * one_vec;
  Eigen::VectorXd tau1 = ((tA - tD) * 1.875).array() / tausmooth.array();
  Eigen::VectorXd tau2 = (tA + tD) * 0.5;


  Eigen::VectorXd gain(model->nu);
  Eigen::VectorXd bias(model->nu);
  Eigen::VectorXd force0(model->nu);
  
  Eigen::VectorXd len_ranges(model->nu);
  
  for (int idx_actuator = 0; idx_actuator < model->nu; ++idx_actuator) {
    double length = data_copy2->actuator_length[idx_actuator];
    mjtNum* lengthrange = (mjtNum*) mju_malloc(2 * sizeof(mjtNum));
    lengthrange[0] = model->actuator_lengthrange[2*idx_actuator];
    lengthrange[1] = model->actuator_lengthrange[2*idx_actuator+1];
    double velocity = data_copy2->actuator_velocity[idx_actuator];
    double acc0 = model->actuator_acc0[idx_actuator];
    mjtNum* prmb = (mjtNum*) mju_malloc(9 * sizeof(mjtNum));
    for (int j = 0; j<9; j++) {
      prmb[j] = model->actuator_biasprm[10*idx_actuator+j];
    }

    mjtNum* prmg = (mjtNum*) mju_malloc(9 * sizeof(mjtNum));
    for (int j = 0; j<9; j++) {
      prmg[j] = model->actuator_gainprm[10*idx_actuator+j];
    }
    
    len_ranges(idx_actuator) = lengthrange[1] - lengthrange[0];
    bias[idx_actuator] = mju_muscleBias(length, lengthrange, acc0, prmb);
    double g = mju_muscleGain(length, velocity, lengthrange, acc0, prmg);
    if (g >= -1) {
      g = -1;
    }
    force0(idx_actuator) = prmg[2];
    gain[idx_actuator] = g;

    //delete pointers
    delete lengthrange;
    delete prmb;
    delete prmg;
  }

  // // for hand task
  Eigen::VectorXd kp;
  Eigen::VectorXd kd;
  if (model->nu < 100) // hand
  {
    kp = Eigen::VectorXd::Ones(model->nu).array() * 1000;
  kd = Eigen::VectorXd::Ones(model->nu).array() * 0;
  }
  else {
    kp = Eigen::VectorXd::Ones(model->nu).array() * 10000; // human
    kd = Eigen::VectorXd::Ones(model->nu).array() * 0;
  }

  Eigen::VectorXd P = Eigen::VectorXd::Zero(model->nu);
  Eigen::VectorXd D = Eigen::VectorXd::Zero(model->nu);
  Eigen::MatrixXd AM = Eigen::Map<Eigen::MatrixXd>(data_copy2->actuator_moment, model->nv, model->nu).array().abs();
  for (int i = 0; i < dim_high_level_action; i++) {
    P += AM.row(includeList[i]) * abs(data_copy2->qpos[includeList[i]]-data_copy->qpos[includeList[i]]);
    D += AM.row(includeList[i]) * mju_exp(-abs(data_copy2->qpos[includeList[i]]-data_copy->qpos[includeList[i]]));
  }
  
  Eigen::VectorXd target_muscle_length = Eigen::Map<Eigen::VectorXd>(data_copy2->actuator_length, model->nu);
  Eigen::VectorXd current_muscle_length = Eigen::Map<Eigen::VectorXd>(data_copy->actuator_length, model->nu);
  Eigen::VectorXd muscle_velocity = Eigen::Map<Eigen::VectorXd>(data_copy->actuator_velocity, model->nu);
  
  Eigen::VectorXd length_diff = target_muscle_length - current_muscle_length;

  Eigen::VectorXd muscle_force = kp.array() * P.array()*(target_muscle_length-current_muscle_length).array()- kd.array()*D.array() * muscle_velocity.array();
  
  Eigen::VectorXd length0 = Eigen::Map<Eigen::VectorXd>(model->actuator_length0, model->nu);
  
  for (int i=0; i<model->nu; i++) {
    if (muscle_force(i) > 0) {
      muscle_force(i) = 0;
    }
    else {
      muscle_force(i) = muscle_force(i);
    }
  }
  if (data_copy->time == 0) {
    Eigen::VectorXd qpos_vec = Eigen::Map<Eigen::VectorXd>(data_copy2->qpos, model->nq);
    Eigen::VectorXd qvel_vec = Eigen::Map<Eigen::VectorXd>(data_copy2->qvel, model->nq);
  }

  Eigen::VectorXd target_act = (muscle_force.array()-bias.array()) / gain.array();

  Eigen::VectorXd b1 = act0.array() + (ts*(one_vec - act0)).array()  / (tau2 + tau1 * (one_vec - act0)).array();
  Eigen::VectorXd b2 = act0.array() - act0.array() * ts / (tau2- tau1 * act0).array();

  for (int i=0; i<model->nu; i++) {
    double ub, lb;
    if (b1[i] > b2[i]) {
      ub = b1[i];
      lb = b2[i];
    }
    else {
      ub = b2[i];
      lb = b1[i];
    }
    if (target_act[i] < lb) {
      target_act[i] = lb;
    }
    if (target_act[i] > ub) {
      target_act[i] = ub;
    }
  }

  Eigen::VectorXd nominator = act0.array() * act0.array() * tau1.array() -
                              act0.array() * tau2.array() +
                              ts * act0.array() -
                              target_act.array() * act0.array() * tau1.array() +
                              target_act.array() * tau2.array();

  Eigen::VectorXd denominator = act0.array() * tau1.array() +
                                ts * one_vec.array() -
                                target_act.array() * tau1.array();
  
  Eigen::VectorXd ctrl_vec = nominator.array() / (denominator.array());

  for (int i = 0; i < model->nu; i++) {
    if (ctrl_vec(i) > 1) {
      ctrl_vec(i) = 1;
    }
    if (ctrl_vec(i) < 0) {
      ctrl_vec(i) = 0;
    }
  }

  // simulate muscle strain
  if (data_copy2->time >= 0) {
    // // gluteus maximus
    // ctrl_vec(14) = 0;
    // ctrl_vec(15) = 0;
    // ctrl_vec(16) = 0;
    // // adductor magnus
    // ctrl_vec(2) = 0;
    // ctrl_vec(3) = 0;
    // ctrl_vec(4) = 0;
    // ctrl_vec(5) = 0;
    // biceps femoris
    ctrl_vec(6) = 0;
    ctrl_vec(7) = 0;
    // gastrocnemius
    ctrl_vec(12) = 0;
    ctrl_vec(13) = 0;
    // semi membranosus
    ctrl_vec(30) = 0;
    // semi tendinosus
    ctrl_vec(31) = 0;
  }

  return ctrl_vec;
}


// get control from mj data
Eigen::VectorXd HierarchicalPDPolicy::get_line_ctrl() const {

  Eigen::VectorXd act0 = Eigen::Map<Eigen::VectorXd>(data_copy->act, model->nu);
  // Eigen::VectorXd one_vec = Eigen::VectorXd::Ones(model->nu);
  // Eigen::VectorXd ctrl0 = Eigen::Map<Eigen::VectorXd>(data_copy2->ctrl, model->nu);
  // std::cout<<"act "<< act0 <<std::endl;
  // std::cout<<"ctrl0 "<< ctrl0 <<std::endl;
  
  // double ts = model->opt.timestep;

  

  // PD control
  Eigen::VectorXd kp_leg = Eigen::VectorXd::Ones(76).array() * 100000;
  Eigen::VectorXd kp_arm = Eigen::VectorXd::Ones(64).array() * 10000000;
  Eigen::VectorXd kp_torso = Eigen::VectorXd::Ones(86).array() * 100000;
  // Eigen::VectorXd kp = Eigen::VectorXd::Ones(model->nu).array() * 10000;
  Eigen::VectorXd kd = Eigen::VectorXd::Ones(model->nu).array() * 0;
  
  // Eigen::VectorXd kd_leg = Eigen::VectorXd::Ones(78).array() * 0;
  // Eigen::VectorXd kd_arm = Eigen::VectorXd::Ones(94).array() * 0;
  // Eigen::VectorXd kd_torso = Eigen::VectorXd::Ones(528).array() * 0;

  // double kp = 400;
  // double kd = 1;
  // double scale = 200;
  Eigen::VectorXd P = Eigen::VectorXd::Zero(model->nu);
  Eigen::VectorXd D = Eigen::VectorXd::Zero(model->nu);
  Eigen::VectorXd kp = Eigen::VectorXd(model->nu);
  kp << kp_leg, kp_arm, kp_torso;
  // Eigen::VectorXd kd = Eigen::VectorXd(model->nu);
  // kd << kd_leg, kd_arm, kd_torso;
  // std::cout<<"P "<<P(0)<<" "<<P.size()<<std::endl;
  Eigen::MatrixXd AM = Eigen::Map<Eigen::MatrixXd>(data_copy2->actuator_moment, model->nv, model->nu).array().abs();
  for (int i = 0; i < dim_high_level_action; i++) {
    // std::cout<<i<<" "<<AM.row(includeList[i]).size()<<" "<<P.size()<<std::endl;
    // Eigen::VectorXd am =  AM.row(includeList[i]).array() * (model->jnt_range[2*includeList[i]+1] - model->jnt_range[2*includeList[i]]);
    // std::cout<<"am "<<am.maxCoeff()<<" "<<am.minCoeff()<<" "<<am.size()<<std::endl;
    P += AM.row(includeList[i]) * abs(data_copy2->qpos[includeList[i]]-data_copy->qpos[includeList[i]]);
    D += AM.row(includeList[i]) * mju_exp(-abs(data_copy2->qpos[includeList[i]]-data_copy->qpos[includeList[i]]));
    //  * (model->jnt_range[2*includeList[i]+1] - model->jnt_range[2*includeList[i]]);
  }
  
  
  // P = P.array() / (P.maxCoeff()+1e-7);
  // std::cout<<"P "<<P(0)<<" "<<P.size()<<" "<<P.maxCoeff()<<" "<<P.minCoeff()<<std::endl;
  // std::cout<<"D "<<D.maxCoeff()<<" "<<D.minCoeff()<<std::endl;
  // D = D.array() / (D.maxCoeff()+1e-7);
  // Eigen::VectorXd P = AM.colwise().sum();
  
  // D = (P.array()*kp.array()).sqrt();
  // std::cout<<D.maxCoeff()<<" "<<D.minCoeff()<<" "<<D.mean()<<std::endl;
  // double kp = 0.05;
  // double kd = 0.01;
  // std::cout<<"force0 "<<force0<<std::endl;
  // muscle force = max(0, kp*(target_length-current_length)-kd*qvel)
  Eigen::VectorXd target_muscle_length = Eigen::Map<Eigen::VectorXd>(data_copy2->actuator_length, model->nu);
  Eigen::VectorXd current_muscle_length = Eigen::Map<Eigen::VectorXd>(data_copy->actuator_length, model->nu);
  Eigen::VectorXd muscle_velocity = Eigen::Map<Eigen::VectorXd>(data_copy->actuator_velocity, model->nu);
  // std::cout<<"current muscle vel"<<muscle_velocity.maxCoeff()<<" "<<muscle_velocity.minCoeff()<<std::endl;
  // std::cout<<"current muscle len "<<current_muscle_length.maxCoeff()<<" "<<current_muscle_length.minCoeff()<<std::endl;
  // std::cout<<kp_vec<<std::endl;
  
  Eigen::VectorXd length_diff = target_muscle_length - current_muscle_length;
  // std::cout<<"muscle len diff "<<length_diff.maxCoeff()<<" "<<length_diff.minCoeff()<<std::endl;

  // Eigen::VectorXd muscle_force = kp*(target_muscle_length-current_muscle_length).array()/len_ranges.array() - kd * muscle_velocity.array()/len_ranges.array();
  // Eigen::VectorXd muscle_force = 1000*(target_muscle_length-current_muscle_length).array();
  
  // Eigen::VectorXd muscle_force = kp * kp_vec.array()*(target_muscle_length-current_muscle_length).array() - kd*kd_vec.array() * muscle_velocity.array();
  Eigen::VectorXd muscle_force = kp.array() * P.array()*(target_muscle_length-current_muscle_length).array()- kd.array()*D.array() * muscle_velocity.array();
  // std::cout<<"muscle force "<<muscle_force.sum()<<P.sum()<<std::endl;
  // double gain_mean = (kp.array() * P.array()).maxCoeff();
  // std::cout<<"gain mean "<<gain_mean<<std::endl;

  
  for (int i=0; i<model->nu; i++) {
    if (muscle_force(i) > 0) {
      muscle_force(i) = 0;
    }
    else {
      // muscle_force(i) = muscle_force(i)*force0(i);
      muscle_force(i) = muscle_force(i);
    }
  }

  // std::cout<<"muscle force "<<muscle_force.sum()<<P.sum()<<std::endl;
  
  Eigen::VectorXd ctrl_vec = muscle_force;

  for (int i = 0; i < model->nu; i++) {
    // ctrl_vec[i] = std::clamp(ctrl_vec[i], 0.0, 1.0);
    if (ctrl_vec(i) > 0) {
      ctrl_vec(i) = 0;
    }
    if (ctrl_vec(i) < -5) {
      ctrl_vec(i) = -5;
    }
  }

  

  
  return ctrl_vec;
}

// get control by position
Eigen::VectorXd HierarchicalPDPolicy::get_ctrl(double* target_qpos) const{
  mju_copy(data_copy2->qvel, target_qpos, model->nq);
  mju_subFrom(data_copy2->qvel, data_copy2->qpos, model->nq);
  mju_scl(data_copy2->qvel, data_copy2->qvel, 1/model->opt.timestep,model->nq);
  mju_copy(data_copy2->qpos, target_qpos, model->nq);
  
  mj_step1(model, data_copy2);

  Eigen::VectorXd ctrl_vec = get_mus_ctrl();
  return ctrl_vec;
}


//get control by velocity
Eigen::VectorXd HierarchicalPDPolicy::get_ctrl2(double* target_qvel) const{



  mju_copy(data_copy2->qvel, target_qvel, model->nq);
  // mju_scl(data_copy2->qvel, data_copy2->qvel, 1/model->opt.timestep, model->nq);
  mju_scl(data_copy2->qpos, data_copy2->qvel, model->opt.timestep, model->nq);
  mju_addTo(data_copy2->qpos, data_copy->qpos, model->nq);
  Clamp(data_copy2->qpos, model->jnt_range, model->nq);
  mj_step1(model, data_copy2);// gain, bias, and moment depend on qpos and qvel
  Eigen::VectorXd ctrl_vec = get_mus_ctrl();
  return ctrl_vec;
}



// copy policy
void HierarchicalPDPolicy::CopyFrom(const HierarchicalPDPolicy& policy, int horizon) {
  this->plan = policy.plan;
  num_spline_points = policy.num_spline_points;
  // data_copy = mj_copyData(model, policy.data_copy);
  // data_copy2 = mj_copyData(model, policy.data_copy2); 
  mju_copy(data_copy->qpos, policy.data_copy->qpos, model->nq);
  mju_copy(data_copy->qvel, policy.data_copy->qvel, model->nv);
  mju_copy(data_copy->act,  policy.data_copy->act,  model->na);
  mju_copy(data_copy->actuator_length, policy.data_copy->actuator_length, model->nu);
  mju_copy(data_copy->actuator_velocity, policy.data_copy->actuator_velocity, model->nu);

  mju_copy(data_copy2->qpos, policy.data_copy2->qpos, model->nq);
  mju_copy(data_copy2->qvel, policy.data_copy2->qvel, model->nv);
  mju_copy(data_copy2->act,  policy.data_copy2->act,  model->na);
  mju_copy(data_copy2->actuator_length, policy.data_copy2->actuator_length, model->nu);
  mju_copy(data_copy2->actuator_velocity, policy.data_copy2->actuator_velocity, model->nu);
  
}

// copy parameters
void HierarchicalPDPolicy::SetPlan(const TimeSpline& plan) {
  this->plan = plan;
}

}  // namespace mjpc
