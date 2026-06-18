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

#include "mjpc/planners/hierarchical_sampling/policy.h"

#include <absl/random/random.h>

#include <absl/log/check.h>
#include <absl/types/span.h>
#include <mujoco/mujoco.h>
#include "mjpc/spline/spline.h"
#include "mjpc/task.h"
#include "mjpc/trajectory.h"
#include "mjpc/utilities.h"
#include <OsqpEigen/OsqpEigen.h>
#include <Eigen/Dense>
#include <time.h>
// #include <glpk.h>


namespace mjpc {

using mjpc::spline::TimeSpline;

// allocate memory
void HierarchicalSamplingPolicy::Allocate(const mjModel* model, const Task& task,
                              int horizon) {
  // model
  this->model = model;

  // spline points
  num_spline_points = GetNumberOrDefault(kMaxTrajectoryHorizon, model,
                                         "sampling_spline_points");
  dim_high_level_action = model->nv;  // number of high-level actions, set to joint number for MS model
  high_level_actions.resize(kMaxTrajectoryHorizon * dim_high_level_action);   // (horizon-1 x num_action)

  // plan = TimeSpline(/*dim=*/model->nu);
  plan = TimeSpline(/*dim=*/dim_high_level_action);
  plan.Reserve(num_spline_points);
  
	// initialize data copy
  data_copy = mj_makeData(model);
	data_copy2 = mj_makeData(model);
//TODO: compute AM inside action
  time_t start = clock();

  
//   // initialize LP matrices
  model_copy = mj_copyModel(NULL, model);
//   mjData* data = mj_makeData(model);
//   mj_forward(model, data);
//   Eigen::MatrixXd AM = Eigen::Map<Eigen::MatrixXd>(data->actuator_moment, model->nv, model->nu);
//   // Eigen::SparseMatrix<double> P(model->nu + model->nv, model->nu + model->nv);
  // P.resize(model->nu + model->nv, model->nu + model->nv);
  // // Eigen::VectorXd q(model->nu + model->nv);

  

  // q = Eigen::VectorXd::Zero(model->nu + model->nv);                 //Q: n*1向量
  // for (int i = 0; i < model->nv; i++) {
  //   q(model->nu + i) = 1;
  // }

  // std::cout<<"q"<<q<<" "<<q.size()<<std::endl;
  
  // for LP
  c = Eigen::VectorXd::Zero(model->nu + model->nv);                 //Q: n*1向量
  for (int i = 0; i < model->nv; i++) {
    c(model->nu + i) = 1;
  }
  

  std::cout<<"get c used "<<(double)(clock() - start)/CLOCKS_PER_SEC<<std::endl;

  // linearMatrix.resize(model->nv*2+model->nu*2, model->nu + model->nv);
  // Eigen::MatrixXd AM = Eigen::MatrixXd::Zero(model->nv, model->nu);
  // Eigen::MatrixXd jointOnes = Eigen::MatrixXd::Ones(model->nv, model->nv);
  // Eigen::MatrixXd jointZeros = Eigen::MatrixXd::Zero(model->nu, model->nv);
  // Eigen::MatrixXd xOnes = Eigen::MatrixXd::Ones(model->nu, model->nu);
  // linearMatrix << AM, -jointOnes, -AM, -jointOnes, -xOnes, jointZeros, xOnes, jointZeros;
  // std::cout<<"get lin used "<<(double)(clock() - start)/CLOCKS_PER_SEC<<std::endl;

}

// reset memory to zeros
void HierarchicalSamplingPolicy::Reset(int horizon, const double* initial_repeated_action) {
  plan.Clear();
  if (initial_repeated_action != nullptr) {
    plan.AddNode(0, absl::MakeConstSpan(initial_repeated_action, dim_high_level_action));
  }
}

// set action from policy
void HierarchicalSamplingPolicy::Action(double* action, const double* state,
                            double time) const {
  CHECK(action != nullptr);
  
  //TODO: sample high-level action and then convert to low-level action
  std::vector<double> high_level_action(dim_high_level_action);
  plan.Sample(time, absl::MakeSpan(&high_level_action[0], dim_high_level_action));
  // HighToLowAction(&high_level_action[0], action, );
  // // Clamp controls
//   Clamp(action, model->actuator_ctrlrange, model->nu);
  Clamp(action, model->jnt_range, model->nq);
}

void HierarchicalSamplingPolicy::HierarchicalAction(double* action, mjData* data) const {
  CHECK(action != nullptr);
  
  //TODO: sample high-level action and then convert to low-level action
  std::vector<double> high_level_action(dim_high_level_action);
  plan.Sample(data->time, absl::MakeSpan(&high_level_action[0], dim_high_level_action));
  HighToLowAction(&high_level_action[0], action, data);
  // Clamp controls
  Clamp(action, model->actuator_ctrlrange, model->nu);
}

void HierarchicalSamplingPolicy::HighToLowAction(double* high_level_action, double* action, mjData* data) const {
  CHECK(action != nullptr);
  // std::cout<<"model para"<<model->nq<<model->nv<<std::endl;
  // for (int k = 0; k < model->nq; k++) {
  //   std::cout<<"before joint "<<k<<": ";
  //   // std::cout<<temp_qpos[k]<<" "<<high_level_action[k]<<" ";

  //   std::cout<<model->jnt_range[k*2]<<" "<<model->jnt_range[k*2+1]<<std::endl;
  //   }
  // std::cout<<std::endl;
  absl::BitGen gen_;
  // for (int k = 0; k < dim_high_level_action; k++) {
  //     // double noise = absl::Gaussian<double>(gen_, 0.0, 0.0005);
  //     high_level_action[k] = 0;
  //   }
  Clamp(high_level_action, model->jnt_range, model->nq);
  // for (int k = 0; k < model->nq; k++) {
  //   std::cout<<high_level_action[k]<<" ";
  //   }
  // std::cout<<std::endl;
  // time_t start = clock();
  // mjModel* model_copy = mj_copyModel(NULL, model);
  
  // mj_copyModel(NULL, model);
  // std::cout<<"copy model used "<<(double)(clock() - start)/CLOCKS_PER_SEC<<std::endl;
  // test for qfrc: pass
  // start = clock();
  // mjData* data_copy = mj_copyData(NULL, model_copy, data);
  
  // // mj_copyData(NULL, model_copy, data);
  // // std::cout<<"copy data used "<<(double)(clock() - start)/CLOCKS_PER_SEC<<std::endl;
  // double* temp_qpos = data->qpos;
  // double jnt_err = 0;
  // for (int k = 0; k < model->nq; k++) {
  //   // std::cout<<"before joint "<<k<<": ";
  //   // std::cout<<k<<" "<<temp_qpos[k]<<" "<<high_level_action[k]<<" "<<model->jnt_range[k*2]<<" "<<model->jnt_range[k*2+1]<<std::endl;

  //   // std::cout<<temp_qpos[k]-high_level_action[k]<<" ";
  //   jnt_err += mju_abs(temp_qpos[k]-high_level_action[k]);
  //   // break;
  //   }
  // std::cout<<"before joint "<<data->time<<" "<<jnt_err<<std::endl;
  

  // for (int k = 0; k < 1; k++) {
  //   double* qfrc_inverse = get_qfrc(model_copy, high_level_action);
  //   data->qfrc_applied = qfrc_inverse;
  //   Eigen::VectorXd qfrc_vec = Eigen::Map<Eigen::VectorXd>(qfrc_inverse, model->nv);
  //   std::cout<<"qfrc_vec "<<qfrc_vec<<std::endl;
  //   // std::cout<<"before time "<<data_copy->time<<std::endl; 
  //   mj_step(model_copy, data);
  //   // std::cout<<"after time "<<data_copy->time<<" "<<model->opt.timestep<<std::endl; 
  //   double* temp_qpos = data->qpos;
  //   // std::cout<<"iteration "<<k<<std::endl;
  //   jnt_err = 0;
  //   for (int j = 0; j < model->nq; j++) {
  //     // std::cout<<"joint "<<j<<": ";
  //     // std::cout<<temp_qpos[j]<<" "<<high_level_action[j]<<std::endl;
  //     jnt_err += mju_abs(temp_qpos[k]-high_level_action[k]);
  //     // break;
  //     }
  //   std::cout<<"iteration "<<k<< " joint "<<jnt_err<<std::endl;
  //   }

    
    
  
  // // for (int k = 0; k < model->nv; k++) {
  // //   std::cout<<qfrc_inverse[k]<<std::endl;
  // //   }

  //test QP
  // solve_qp();

  //test ctrl
  // start = clock();
  double* qfrc_inverse = get_qfrc(model_copy, high_level_action);
  // std::cout<<"get force used "<<(double)(clock() - start)/CLOCKS_PER_SEC<<std::endl;



  // check control
  // for (int i = 0; i < model->nu; i++) {
  //   std::cout << "origin ctrl[" << i << "]: " << data->ctrl[i] << std::endl;
  //   std::cout << "copy ctrl[" << i << "]: " << data_copy2->ctrl[i] << std::endl;
  // }
  
  // start = clock();
  // // std::cout<<"test "<<(double)(clock() - start)/CLOCKS_PER_SEC<<std::endl;
  Eigen::VectorXd action_vec = get_ctrl(high_level_action, qfrc_inverse);
  // std::cout<<"get ctrl used "<<(double)(clock() - start)/CLOCKS_PER_SEC<<std::endl;

  // std::cout<<"finish id"<<std::endl;
  // double* qfrc_inverse = get_qfrc(model_copy, data_copy, high_level_action);
  // // std::cout<<"qfrc_inverse "<< qfrc_inverse[0] <<std::endl;
  // double* ctrl = get_ctrl(model_copy, data, high_level_action);

  
  // mju_error_i(
  //       "run end here", 0
  //       );
  
  

  //add random noise for test
  // sampling token
  // absl::BitGen gen_;
  for (int k = 0; k < model->nu; k++) {
      // double noise = absl::Gaussian<double>(gen_, 0.0, 0.005);
      action[k] =action_vec(k);
    }
  
  Clamp(action, model->actuator_ctrlrange, model->nu);
  // std::cout<<"finish ctrl"<<Eigen::Map<Eigen::VectorXd>(action, model->nu) <<std::endl;
  // mju_error_i(
  //       "run end here", 0
  //       );

}


// // Solve a quadratic program
Eigen::VectorXd HierarchicalSamplingPolicy::solve_qp(
    Eigen::SparseMatrix<double> hessian, 
  Eigen::VectorXd gradient, 
  Eigen::SparseMatrix<double> constraintMatrix,
  Eigen::VectorXd lowerBound,
  Eigen::VectorXd upperBound,
  Eigen::VectorXd initialGuess
) const{

  //test QP
  // allocate QP problem matrices and vectores
  // Eigen::SparseMatrix<double> hessian2(2, 2);      //P: n*n正定矩阵,必须为稀疏矩阵SparseMatrix
  // Eigen::VectorXd gradient2(2);                    //Q: n*1向量
  

  
  
  // Eigen::VectorXd lowerBound2(2);                  //L: m*1下限向量
  // Eigen::VectorXd upperBound2(2);                  //U: m*1上限向量
  // lowerBound2 << 0, 0;
  // upperBound2 << 1.5, 1.5;
  int m = lowerBound.size();
  int n = gradient.size();
  // std::cout<<"m "<<m<<" n "<<n<<std::endl;
  
  // // std::cout << "size" << hessian.rows() << "x" << hessian.cols() << " " << lowerBound.size() << std::endl;
  // hessian2.insert(0, 0) = 2.0; //注意稀疏矩阵的初始化方式,无法使用<<初始化
  // hessian2.insert(1, 1) = 2.0;
  // // std::cout << "hessian:" << std::endl
  // //           << hessian << std::endl;
  // gradient2 << -2, -2;

  // Eigen::SparseMatrix<double> linearMatrix(m, n); //A: m*n矩阵,必须为稀疏矩阵SparseMatrix
  // for (int i = 0; i < n; i++) {
  //   linearMatrix.insert(i, i) = 1.0;
  // }
  // linearMatrix.insert(0, 0) = 1.0; //注意稀疏矩阵的初始化方式,无法使用<<初始化
  // linearMatrix.insert(1, 1) = 1.0;
  // std::cout << "linearMatrix:" << std::endl
  //           << linearMatrix << std::endl;
  

  // instantiate the solver
  OsqpEigen::Solver* solver = new OsqpEigen::Solver();

  // solver.settings()->setVerbosity(true);
  // solver.settings()->setWarmStart(true);

  // settings
  solver->settings()->setVerbosity(false);
  solver->settings()->setWarmStart(true);
  
  // set the initial data of the QP solver
  solver->data()->setNumberOfVariables(n);   //变量数n
  solver->data()->setNumberOfConstraints(m); //约束数m
  


  if(!solver->data()->setHessianMatrix(hessian)){
        std::cout << "H matrix error!" << std::endl;
    } 
  if(!solver->data()->setGradient(gradient)){
      std::cout << "f matrix error!" << std::endl;
  }
  if(!solver->data()->setLinearConstraintsMatrix(constraintMatrix)){
      std::cout << "A matrix error!" << std::endl;
  }
  if(!solver->data()->setLowerBound(lowerBound)){
      std::cout << "lb matrix error!" << std::endl;
  }
  if(!solver->data()->setUpperBound(upperBound)){
      std::cout << "ub matrix error!" << std::endl;
  }


  // initial the solver
  if(!solver->initSolver()){
      std::cout << "solver initial error!" << std::endl;
  }

  // solver.data()->setHessianMatrix(hessian2);
  // solver.data()->setGradient(gradient2);
  // solver.data()->setLinearConstraintsMatrix(linearMatrix);
  // solver.data()->setLowerBound(lowerBound2);
  // solver.data()->setUpperBound(upperBound2);

  // solver.initSolver();
  solver->setPrimalVariable(initialGuess);
  
  // std::cout << "begin res " << initialGuess(0, 0) << std::endl;
  // Eigen::VectorXd QPSolution;
  // try {
  //   if (solver.solveProblem()!= OsqpEigen::ErrorExitFlag::NoError) {
  //     std::cout << "QP not solve" << std::endl;
  //     QPSolution = Eigen::VectorXd::Zero(n);
  //   }
  //   else {
  //     std::cout << "QP solved" << std::endl;
  //     QPSolution = solver.getSolution();
  //   }
  // }
  // catch (...) {
  //   std::cout << "QP not solve err" << std::endl;
  // }
  // time_t start = clock();
  solver->solveProblem();
  // std::cout<<"solve problem used "<<(double)(clock() - start)/CLOCKS_PER_SEC<<std::endl;
  // std::cout<<"solve res "<< solveres << std::endl;  
  Eigen::VectorXd QPSolution = solver->getSolution();
  // solver->getSolution();
  // solver.updateBounds(lowerBound, upperBound)
  // Eigen::VectorXd QPSolution = solver.m_solution;
  
  solver->clearSolverVariables();
  // solver.initSolver();
  // OsqpEigen::Solver::OSQPSolverDeleter(solver);
  // TODO: find a better way to delete solver
  solver = new OsqpEigen::Solver();
  // std::cout << "QPSolution " << QPSolution << std::endl; 
  // delete solver;
  // solver.~Solver();
  // std::cout<<"before return"<<std::endl;
  return QPSolution;
}


Eigen::VectorXd HierarchicalSamplingPolicy::linprog(
  Eigen::VectorXd c,
  Eigen::MatrixXd A,
  Eigen::VectorXd b,
  Eigen::VectorXd x0) const{

  bool ipm = false;
  bool verbose = false;
  
  int d = c.size();
  int m = b.size();
  int dm = d * m;
  // Eigen::VectorXd x(d);
  // x << x0;
  Eigen::VectorXd x = Eigen::VectorXd::Zero(d);
  glp_prob *lp;
  int *ia = new int[dm + 1];
  int *ja = new int[dm + 1];
  double *ar = new double[dm + 1];
  int s;
  double z;

  lp = glp_create_prob();
  glp_set_prob_name(lp, "lp");
  glp_set_obj_dir(lp, GLP_MIN);

  glp_add_rows(lp, m);
  for (int i = 1; i <= m; i++)
  {
      glp_set_row_name(lp, i, (std::to_string(i) + "y").c_str());
      glp_set_row_bnds(lp, i, GLP_UP, 0.0, b(i - 1));
  }

  glp_add_cols(lp, d);
  for (int i = 1; i <= d; i++)
  {
      glp_set_col_name(lp, i, (std::to_string(i) + "x").c_str());
      glp_set_col_bnds(lp, i, GLP_FR, 0.0, 0.0);
      glp_set_obj_coef(lp, i, c(i - 1));
  }

  int k = 1;
  for (int i = 1; i <= m; i++)
  {
      for (int j = 1; j <= d; j++)
      {
          ia[k] = i;
          ja[k] = j;
          ar[k] = A(i - 1, j - 1);
          k++;
      }
  }
  glp_load_matrix(lp, dm, ia, ja, ar);

  if (!ipm)
  {
      glp_smcp param;
      glp_init_smcp(&param);
      param.msg_lev = verbose ? GLP_MSG_ALL : GLP_MSG_OFF;
      glp_simplex(lp, &param);
      s = glp_get_status(lp);
      z = INFINITY;
      if (s == GLP_OPT || s == GLP_UNBND)
      {
          z = (s == GLP_UNBND) ? -INFINITY : glp_get_obj_val(lp);
          for (int i = 1; i <= d; i++)
          {
              x(i - 1) = glp_get_col_prim(lp, i);
          }
      }
  }
  else
  {
      glp_iptcp param;
      glp_init_iptcp(&param);
      param.msg_lev = verbose ? GLP_MSG_ALL : GLP_MSG_OFF;
      glp_interior(lp, &param);
      s = glp_ipt_status(lp);
      z = INFINITY;
      if (s == GLP_OPT)
      {
          z = glp_ipt_obj_val(lp);
          for (int i = 1; i <= d; i++)
          {
              x(i - 1) = glp_ipt_col_prim(lp, i);
          }
      }
  }

  glp_delete_prob(lp);
  delete[] ia;
  delete[] ja;
  delete[] ar;
  // std::cout<<"linprog res "<<z<<std::endl;
  // return z;
  //use z
  z++;

  return x;
  }

// get qfrc
double* HierarchicalSamplingPolicy::get_qfrc(mjModel* model, double* target_qpos) const{
// Compute the generalized force needed to reach the target position in the next mujoco step
  // std::cout<<"begin get_qfrc"<<std::endl;
  // mjData* data_copy = mj_copyData(NULL, model, data);
  // mjData* data_copy = data;
  // std::unique_ptr<mjData> data_copy = mj_copyData(NULL, model, data);
  // mjData* data_copy = mj_makeData(model);
  // mj_forward(model, data_copy);
  // copy qpos, qvel, mocap qpos, mocap_quat from original data
  // mju_copy(data_copy->qpos, data->qpos, model->nq);
  // mju_copy(data_copy->qvel, data->qvel, model->nv);
  // mju_copy(data_copy->mocap_pos, data->mocap_pos, model->nmocap * 3);
  // mju_copy(data_copy->mocap_quat, data->mocap_quat, model->nmocap * 4);

  // mjtNum* target_qacc = nullptr;
  // mjtNum* target_qacc = (mjtNum*) mju_malloc(model->nq * sizeof(mjtNum));
  // std::cout<<"copy acc"<<std::endl;

  mju_copy(data_copy->qacc, target_qpos, model->nq);

  
  // memcpy(&target_qacc[0], &target_qpos[0], model->nq * sizeof(double));
  //velocity
  // std::cout<<"get acc"<<std::endl;
  mju_subFrom(data_copy->qacc, data_copy->qpos, model->nq);
  // for (int k = 0; k < model->nv; k++) {
  //   std::cout<<"joint "<<k<<": ";
  //   std::cout<<target_qacc[k]<<" ";
  //   }
  // std::cout<<std::endl;
  mju_scl(data_copy->qacc, data_copy->qacc, 1/model->opt.timestep, model->nq);
  // for (int k = 0; k < model->nv; k++) {
  //   std::cout<<"joint "<<k<<": ";
  //   std::cout<<target_qacc[k]<<" ";
  //   }
  // std::cout<<std::endl;
  
  //acc
  mju_subFrom(data_copy->qacc, data_copy->qvel, model->nq);
  mju_scl(data_copy->qacc, data_copy->qacc, 1/model->opt.timestep, model->nq);
  
  // for (int k = 0; k < model->nv; k++) {
  //   std::cout<<"joint "<<k<<": ";
  //   std::cout<<target_qacc[k]<<" ";
  //   }
  // std::cout<<std::endl;
  // data_copy->qacc = target_qacc;
  //inverse dynamics
  // std::cout<<"begin inverse dynamics"<<std::endl;
  model->opt.disableflags += mjDSBL_CONSTRAINT;
  // std::cout<<"begin mj_inverse"<<std::endl;
  mj_inverse(model, data_copy);
  // std::cout<<"end mj_inverse"<<std::endl;
  model->opt.disableflags -= mjDSBL_CONSTRAINT;
  double* qfrc_inverse = data_copy->qfrc_inverse;

  // for (int k = 0; k < model->nv; k++) {
  //   std::cout<<"joint force "<<k<<": ";
  //   std::cout<<qfrc_inverse[k]<<" ";
  //   }
  // std::cout<<std::endl;
  
  
  // delete data_copy;
  //clamp too large force
  // for (int k = 0; k < model->nv; k++) {
  //   if (qfrc_inverse[k] > 1000) {
  //     qfrc_inverse[k] = 1000;
  //   }
  //   if (qfrc_inverse[k] < -1000) {
  //     qfrc_inverse[k] = -1000;
  //   }
  // }


  return qfrc_inverse;
}

// get control
Eigen::VectorXd HierarchicalSamplingPolicy::get_ctrl(double* target_qpos, double* qfrc) const{
  double qfrc_scaler = 1;
  double qvel_scaler = 1;
  // time_t start = clock();
  // Compute the control needed to reach the target position in the next mujoco step
  Eigen::VectorXd act_vec = Eigen::Map<Eigen::VectorXd>(data_copy2->act, model->nu);
  Eigen::VectorXd ctrl0 = Eigen::Map<Eigen::VectorXd>(data_copy2->ctrl, model->nu);
  // std::cout<<"act "<< act <<std::endl;
  // std::cout<<"ctrl0 "<< ctrl0 <<std::endl;

  double ts = model->opt.timestep;
  Eigen::VectorXd tA = 0.01 * (0.5 + 1.5 * act_vec.array());
  Eigen::VectorXd tD = 0.04 / (0.5 + 1.5 * act_vec.array());
  Eigen::VectorXd tausmooth = 5 * Eigen::VectorXd::Ones(model->nu);
  Eigen::VectorXd t1 = ((tA - tD) * 1.875).array() / tausmooth.array();
  Eigen::VectorXd t2 = (tA + tD) * 0.5;

  // ---- gain, bias, and moment computation
  // mjData* data_copy = mj_copyData(NULL, model, data);
  
  // mjtNum* qvel = (mjtNum*) mju_malloc(model->nq * sizeof(mjtNum));
  mju_copy(data_copy2->qvel, target_qpos, model->nq);
  mju_subFrom(data_copy2->qvel, data_copy2->qpos, model->nq);
  mju_scl(data_copy2->qvel, data_copy2->qvel, 1/model->opt.timestep, model->nq);
  mju_scl(data_copy2->qvel, data_copy2->qvel, 1/qvel_scaler, model->nq);

  mju_copy(data_copy2->qpos, target_qpos, model->nq);
  std::cout<<"qpos "<<data_copy2->qpos[0]<<std::endl;
  std::cout<<"qvel "<<data_copy2->qvel[0]<<std::endl;
  // data_copy->qvel= qvel; 
  mj_step1(model, data_copy2); // gain, bias, and moment depend on qpos and qvel
  Eigen::VectorXd gain(model->nu);
  Eigen::VectorXd bias(model->nu);
  
  for (int idx_actuator = 0; idx_actuator < model->nu; ++idx_actuator) {
    double length = data_copy->actuator_length[idx_actuator];
    mjtNum* lengthrange = (mjtNum*) mju_malloc(2 * sizeof(mjtNum));
    lengthrange[0] = model->actuator_lengthrange[2*idx_actuator];
    lengthrange[1] = model->actuator_lengthrange[2*idx_actuator+1];
    double velocity = data_copy->actuator_velocity[idx_actuator];
    double acc0 = model->actuator_acc0[idx_actuator];
    // if (idx_actuator == 0) {
    //   std::cout<<"length "<<length<<std::endl;
    //   std::cout<<"lengthrange "<<lengthrange[0]<<" "<<lengthrange[1]<<std::endl;
    //   std::cout<<"velocity "<<velocity<<std::endl;
    //   std::cout<<"acc0 "<<acc0<<std::endl;
    // }
    mjtNum* prmb = (mjtNum*) mju_malloc(9 * sizeof(mjtNum));
    // Eigen::VectorXd prmb(9);
    for (int j = 0; j<9; j++) {
      prmb[j] = model->actuator_biasprm[10*idx_actuator+j];
    // std::cout<<"biasprm "<<j<<" "<<prmb[j]<<std::endl;
    }
    // std::cout<<std::endl;
    // Eigen::VectorXd prmg(9);
    mjtNum* prmg = (mjtNum*) mju_malloc(9 * sizeof(mjtNum));
    for (int j = 0; j<9; j++) {
      prmg[j] = model->actuator_gainprm[10*idx_actuator+j];
    // std::cout<<"gainprm "<<j<<" "<<prmg[j]<<std::endl;
    }
    // std::cout<<std::endl;
    bias[idx_actuator] = mju_muscleBias(length, lengthrange, acc0, prmb);
    gain[idx_actuator] = std::min(-1.0, mju_muscleGain(length, velocity, lengthrange, acc0, prmg));

    //delete pointers
    delete lengthrange;
    delete prmb;
    delete prmg;
  }

  // std::cout<<"bias "<< bias <<std::endl;
  // std::cout<<"gain "<< gain <<std::endl;
  // std::cout<<"get gains used "<<(double)(clock() - start)/CLOCKS_PER_SEC<<std::endl;

  // start = clock();


  Eigen::MatrixXd AM = Eigen::Map<Eigen::MatrixXd>(data_copy->actuator_moment, model->nv, model->nu);

  // ---- ctrl computation
  // start = clock();

  //formulate as QP
  // Eigen::MatrixXd P = 2 * AM.transpose() * AM;
  // std::cout<<"get P used "<<(double)(clock() - start)/CLOCKS_PER_SEC<<std::endl;
  Eigen::VectorXd qfrc_vec = Eigen::Map<Eigen::VectorXd>(qfrc, model->nv);
  // std::cout<<"qfrc_vec "<<qfrc_vec<<std::endl;
  // std::cout<<"AM shape "<<AM.rows()<<" "<<AM.cols()<<std::endl;
  // std::cout<<"vac shape "<<(gain.array() * act_vec.array()).rows()<<" "<<(gain.array() * act_vec.array()).cols()<<std::endl;

  // std::cout<<(gain.array() * act_vec.array()).size()<<std::endl;

  
  Eigen::VectorXd gain_act = (gain.array() * act_vec.array());

  // start = clock();
  Eigen::VectorXd k =  AM * gain_act + AM * bias - (qfrc_vec/qfrc_scaler);
  // std::cout<<"get vecs used "<<(double)(clock() - start)/CLOCKS_PER_SEC<<std::endl;

  // Eigen::VectorXd q = 2 * k.transpose() * AM;
  // Eigen::VectorXd lb = gain.array() * (1 - act_vec.array()) * ts / (t2.array() + t1.array() * (1 - act_vec.array()));
  // Eigen::VectorXd ub = - gain.array() * act_vec.array() * ts / (t2.array() - t1.array() * act_vec.array());
  // Eigen::VectorXd x0 = (gain.array() * (ctrl0.array() - act_vec.array()) * ts) / ((ctrl0.array() - act_vec.array()) * t1.array() + t2.array());
  
  // Eigen::SparseMatrix<double> P_sparse = P.sparseView();
  // std::cout<<"get matrices used "<<(double)(clock() - start)/CLOCKS_PER_SEC<<std::endl;

  // start = clock();
  // Eigen::VectorXd x = solve_qp(P_sparse, q, lb, ub, x0);
  // std::cout<<"get qp used"<<(double)(clock() - start)/CLOCKS_PER_SEC<<std::endl;
  // Eigen::VectorXd x = Eigen::VectorXd::Zero(model->nu); 



  // formulate as LP
  //use axullury variable z to convert to LP

  // Eigen::SparseMatrix<double> P(model->nu + model->nv, model->nu + model->nv);      //P: n*n正定矩阵,必须为稀疏矩阵SparseMatrix

  // std::cout<<"get P used "<<(double)(clock() - start)/CLOCKS_PER_SEC<<std::endl;

  // start = clock();
  // //minimize sum of z
  // only initialize once as the matrices are fixed
  // if (initialized == false) {
  //   Eigen::SparseMatrix<double> P(model->nu + model->nv, model->nu + model->nv);
  //   Eigen::VectorXd q(model->nu + model->nv);


  //   q = Eigen::VectorXd::Zero(model->nu + model->nv);                 //Q: n*1向量
  //   for (int i = 0; i < model->nv; i++) {
  //     q(model->nu + i) = 1;
  //   }

  //   std::cout<<"get q used "<<(double)(clock() - start)/CLOCKS_PER_SEC<<std::endl;

  //   start = clock();

  //   // Eigen::MatrixXd linearMatrix(model->nv*2+model->nu, model->nu + model->nv); //A: m*n矩阵,必须为稀疏矩阵SparseMatrix
  //   // Eigen::MatrixXd jointOnes = Eigen::MatrixXd::Ones(model->nv, model->nv);
  //   // Eigen::MatrixXd jointZeros = Eigen::MatrixXd::Zero(model->nu, model->nv);
    
  //   // Eigen::MatrixXd xOnes = Eigen::MatrixXd::Ones(model->nu, model->nu);

  //   // Eigen::MatrixXd cons1(model->nv, model->nu+model->nv);
  //   // cons1 << AM, jointOnes;
  //   // std::cout<<"con1 finish"<<std::endl;
  //   // Eigen::MatrixXd cons2(model->nv, model->nu+model->nv);
  //   // cons2 << AM, jointOnes*-1;
  //   // std::cout<<"con2 finish"<<std::endl;
  //   // Eigen::MatrixXd cons3(model->nu, model->nu+model->nv);
  //   // // std::cout<<cons3.rows()<<" "<<cons3.cols()<<" "<<xOnes.rows()<<" "<<xOnes.cols()<<" "<<jointZeros.rows()<<" "<<jointZeros.cols()<<" "<<std::endl;
  //   // cons3 << xOnes, jointZeros;
  //   // std::cout<<"con3 finish"<<std::endl;

  //   // linearMatrix << cons1, cons2, cons3;
  //   // Eigen::SparseMatrix<double> linearMatrix_sparse = linearMatrix.sparseView();



  // *******************************************QP*******************************************

  // Eigen::SparseMatrix<double> linearMatrix(model->nv*2+model->nu, model->nu + model->nv);

  // //AM * x + z >= -k
  // for (int i = 0; i < model->nv; i++) {
  //   for (int j = 0; j < model->nu; j++) {
  //     if (AM(i, j) > 0) {
  //       linearMatrix.insert(i, j) = AM(i, j);
  //     } 
  //   }
  //   linearMatrix.insert(i, model->nu + i) = 1;
  // }

  


  // // AM * x - z <= -k
  // int nonzero_count = 0;

  // for (int i = 0; i < model->nv; i++) {
  //   for (int j = 0; j < model->nu; j++) {
  //     if (AM(i, j) > 0) {
  //       linearMatrix.insert(model->nv + i, j) = -AM(i, j);
  //       nonzero_count += 1;
  //     } 
  //   }
  //   linearMatrix.insert(model->nv + i, model->nu + i) = -1;
  // }
  // std::cout<<"nonzero count "<<nonzero_count<<std::endl;

  // // lb <= x <= ub
  // for (int i = 0; i < model->nu; i++) {
  //   linearMatrix.insert(model->nv*2 + i, i) = 1;
  // }

  // // std::cout<<"lin block 1"<<linearMatrix.block(0, 0, model->nv, model->nu + model->nv)<<std::endl;
  // // std::cout<<"get lin used "<<(double)(clock() - start)/CLOCKS_PER_SEC<<std::endl;

  // //   initialized = true;
  // // }
  

  // start = clock();
  
  // // get ub and lb
  // Eigen::VectorXd gain_act = (gain.array() * act_vec.array());
  // Eigen::VectorXd k =  AM * gain_act + AM * bias - (qfrc_vec/qfrc_scaler);

  // Eigen::VectorXd lowerBound(model->nv*2 + model->nu);
  // Eigen::VectorXd upperBound(model->nv*2 + model->nu);

  // for (int i = 0; i < model->nv; i++) {
  //   lowerBound(i) = -k(i);
  //   upperBound(i) = OsqpEigen::INFTY;
  // }

  // for (int i = 0; i < model->nv; i++) {
  //   lowerBound(model->nv + i) = -OsqpEigen::INFTY;
  //   upperBound(model->nv + i) = -k(i);
  // }

  // Eigen::VectorXd lb = gain.array() * (1 - act_vec.array()) * ts / (t2.array() + t1.array() * (1 - act_vec.array()));
  // Eigen::VectorXd ub = - gain.array() * act_vec.array() * ts / (t2.array() - t1.array() * act_vec.array());

  // for (int i = 0; i < model->nu; i++) {
  //   lowerBound(model->nv*2 + i) = lb(i);
  //   upperBound(model->nv*2 + i) = ub(i);
  // }

  // // std::cout<<"get bound used "<<(double)(clock() - start)/CLOCKS_PER_SEC<<std::endl;

  // start = clock();


  
  // Eigen::VectorXd full_x0 = Eigen::VectorXd::Zero(model->nu + model->nv);

  // Eigen::VectorXd x0 = (gain.array() * (ctrl0.array() - act_vec.array()) * ts) / ((ctrl0.array() - act_vec.array()) * t1.array() + t2.array());
  

  // for (int i = 0; i < model->nu; i++) {
  //   full_x0(i) = x0(i);
  // }

  // for (int i = 0; i < model->nv; i++) {
  //   full_x0(model->nu + i) = 0;
  // }


  // // std::cout<<"get matrices used "<<(double)(clock() - start)/CLOCKS_PER_SEC<<std::endl;

  // start = clock();

  // // std::cout<<"linear shape"<<linearMatrix.rows()<<" "<<linearMatrix.cols()<<std::endl;
  // // std::cout<<"lower shape"<<lowerBound.size()<<std::endl;
  // // std::cout<<"P shape"<<model->nu + model->nv<<" "<<model->nu + model->nv<<std::endl;

  // Eigen::VectorXd full_x = solve_qp(P, q, linearMatrix, lowerBound, upperBound, full_x0);
  // std::cout<<"get qp used"<<(double)(clock() - start)/CLOCKS_PER_SEC<<std::endl;


  //****************LP*******************

  //linear matrix
  // start = clock();
  // TODO: only update AM part since most blocks are the same
  // Eigen::MatrixXd linearMatrix(model->nv*3+model->nu*2, model->nu + model->nv);
  Eigen::MatrixXd linearMatrix(model->nv*3, model->nu + model->nv);
  Eigen::MatrixXd jointOnes = Eigen::MatrixXd::Identity(model->nv, model->nv);
  Eigen::MatrixXd jointZeros = Eigen::MatrixXd::Zero(model->nu, model->nv);
  Eigen::MatrixXd xOnes = Eigen::MatrixXd::Identity(model->nu, model->nu);
  Eigen::MatrixXd xZeros = Eigen::MatrixXd::Zero(model->nv, model->nu);
  Eigen::MatrixXd JointLim = Eigen::MatrixXd::Identity(model->nv, model->nv);
  // linearMatrix << AM, -jointOnes, -AM, -jointOnes, -xOnes, jointZeros, xOnes, jointZeros, xZeros, -JointLim;
   linearMatrix << AM, -jointOnes, -AM, -jointOnes, xZeros, -JointLim;
  // std::cout<<"lin block 1"<<linearMatrix.block(0, 0, model->nv, model->nu + model->nv)<<std::endl;
  // std::cout<<"lin block 2"<<linearMatrix.block(model->nv, 0, model->nv, model->nu + model->nv)<<std::endl;
  // std::cout<<"lin block 3"<<linearMatrix.block(2*model->nv, 0, model->nv, model->nu + model->nv)<<std::endl;
  // std::cout<<"get lin used "<<(double)(clock() - start)/CLOCKS_PER_SEC<<std::endl;

  // start = clock();
  //bias matrix
  // Eigen::VectorXd b = Eigen::VectorXd::Zero(model->nv*3 + model->nu*2);
  Eigen::VectorXd b = Eigen::VectorXd::Zero(model->nv*3);
  Eigen::VectorXd lb = gain.array() * (1 - act_vec.array()) * ts / (t2.array() + t1.array() * (1 - act_vec.array()));
  Eigen::VectorXd ub = - gain.array() * act_vec.array() * ts / (t2.array() - t1.array() * act_vec.array());

  for (int i = 0; i < model->nv; i++) {
    b(i) = -k(i);
  }
  for (int i = 0; i < model->nv; i++) {
    b(model->nv + i) = k(i);
  }
  // for (int i = 0; i < model->nu; i++) {
  //   // b(2*model->nv + i) = -lb(i);
  //   b(2*model->nv + i) = 1.0e9;
  // }
  // for (int i = 0; i < model->nu; i++) {
  //   // b(2*model->nv + model->nu + i) = ub(i);
  //   b(2*model->nv + model->nu + i) = 1.0e9;
  // }
  for (int i = 0; i < model->nv; i++) {
    b(2*model->nv + 2*model->nu + i) = 0;
  }
  // std::cout<<"gain "<<qfrc_vec<<std::endl;
  // std::cout<<"lb "<<lb<<std::endl;
  // std::cout<<"ub "<<ub<<std::endl;
  // std::cout<<"get bias used "<<(double)(clock() - start)/CLOCKS_PER_SEC<<std::endl;
  


  Eigen::VectorXd full_x0 = Eigen::VectorXd::Zero(model->nu + model->nv);

  Eigen::VectorXd x0 = (gain.array() * (ctrl0.array() - act_vec.array()) * ts) / ((ctrl0.array() - act_vec.array()) * t1.array() + t2.array());
  

  for (int i = 0; i < model->nu; i++) {
    full_x0(i) = x0(i);
  }

  for (int i = 0; i < model->nv; i++) {
    full_x0(model->nu + i) = 100;
  }
  // start = clock();
  Eigen::VectorXd full_x = linprog(c, linearMatrix, b, x0);
  // std::cout<<"get lp used "<<(double)(clock() - start)/CLOCKS_PER_SEC<<std::endl;


  //get ctrl

  // Eigen::VectorXd full_x = Eigen::VectorXd::Zero(model->nu + model->nv);
  //sum of z
  // Eigen::VectorXd z = full_x.tail(model->nv);
  // double z_sum = 0;
  // for (int i = 0; i < model->nv; i++) {
  //   z_sum += z(i);
  // }
  // std::cout<<"z sum "<<z_sum<<std::endl;

  Eigen::VectorXd x = full_x.head(model->nu);
  // std::cout<<"x "<<x<<std::endl; 
  // //test solution
  Eigen::VectorXd res = AM * full_x.head(model->nu);
  // int match_num = 0;
  // for (int i = 0; i < model->nv; i++) {
  //   // std::cout<<"res "<<i<<" "<<res(i)<<" "<<k(i)<<std::endl;
  //   if (abs(res(i) + k(i)) < 0.0001) {
  //     match_num += 1;
  //   }

  // }
  // std::cout<<"match num "<<match_num<<std::endl;

 
  //clamp with lb and ub
  for (int i = 0; i < model->nu; i++) {
    if (x(i)>ub(i)){
      x(i) = ub(i);
    }
    if (x(i)<lb(i)){
      x(i) = lb(i);
    }
  }
  

  Eigen::VectorXd ctrl_vec = act_vec.array() + x.array() * t2.array() / ((gain.array() * ts) - x.array() * t1.array());
  // Eigen::VectorXd ctrl_vec = act_vec.array() / ((gain.array() * ts));

  // for (int i = 0; i < ctrl.size(); ++i) {
  // ctrl[i] = std::clamp(ctrl[i], 0.0, 1.0);
  // }
  // return ctrl;

  // std::cout<<AM.size()<<P.size()<<q.size()<<lb.size()<<x.size()<<ctrl2.size()<<std::endl;

  // double* ctrl = ctrl_vec.data();
  // std::cout<<"control "<<ctrl_vec.sum()<<std::endl;
  //  mju_error_i(
  //       "run end here", 0
  //       );
  // for (int i = 0; i < model->nu; i++) {
  //   std::cout<<"ctrl "<<i<<" "<<ctrl[i]<<std::endl;
  // }
  // std::cout<<"finish get ctrl"<<std::endl;
  return ctrl_vec;
}


// double* HierarchicalSamplingPolicy::get_ctrl(mjModel* model, mjData* data, double* target_qpos, double* qfrc) const{
//   double* act = data->act;
//   // double* ctrl0 = data->ctrl;
//   // double ts = model->opt.timestep;
//   // double qfrc_scaler = 100;
//   double qvel_scaler = 5;

//   mjtNum* tA = (mjtNum*) mju_malloc(model->nu * sizeof(mjtNum));
//   mjtNum* tD = (mjtNum*) mju_malloc(model->nu * sizeof(mjtNum));
//   mjtNum* t1 = (mjtNum*) mju_malloc(model->nu * sizeof(mjtNum));
//   mjtNum* t2 = (mjtNum*) mju_malloc(model->nu * sizeof(mjtNum));
//   double tau_smooth = 5;
//   mjtNum* t_base = (mjtNum*) mju_malloc(model->nu * sizeof(mjtNum));
//   //active
//   mju_zero(t_base, model->nu);
//   mju_copy(t_base, act, model->nu);
//   mju_scl(t_base, t_base, 1.5, model->nu);
//   for (int i = 0; i < model->nu; i++) {
//     t_base[i] += 0.5;
//   }

//   mju_copy(tA, t_base, model->nu);
//   mju_scl(tA, tA, 0.01, model->nu);
//   mju_zero(tD, model->nu);
//   for (int i = 0; i < model->nu; i++) {
//     tD[i] = 0.04/t_base[i];
//   }
  
//   mju_copy(t1, tA, model->nu);
//   mju_subFrom(t1, tD, model->nu);
//   mju_scl(t1, t1, 1.875, model->nu);
//   mju_scl(t1, t1, 1/tau_smooth, model->nu);

//   mju_copy(t2, tA, model->nu);
//   mju_addTo(t2, tD, model->nu);
//   mju_scl(t2, t2, 0.5, model->nu);

//   // gain, bias, moment computation
//   mjData* data_copy = mj_copyData(NULL, model, data);
//   data_copy->qpos = target_qpos;
//   mjtNum* qvel = (mjtNum*) mju_malloc(model->nq * sizeof(mjtNum));
//   mju_copy(qvel, target_qpos, model->nq);
//   mju_subFrom(qvel, data->qpos, model->nq);
//   mju_scl(qvel, qvel, 1/model->opt.timestep, model->nq);
//   mju_scl(qvel, qvel, 1/qvel_scaler, model->nq);
//   data_copy->qvel= qvel; 
//   mj_step1(model, data_copy);

//   mjtNum* gain = (mjtNum*) mju_malloc(model->nu * sizeof(mjtNum));
//   mjtNum* bias = (mjtNum*) mju_malloc(model->nu * sizeof(mjtNum));
//   mju_zero(gain, model->nu);
//   mju_zero(bias, model->nu);

//   for (int i = 0; i < model->nu; i++) {
//     double length = data_copy->actuator_length[i];
//     mjtNum* lengthange = (mjtNum*) mju_malloc(2 * sizeof(mjtNum));
//     lengthange[0] = model->actuator_lengthrange[2*i];
//     lengthange[1] = model->actuator_lengthrange[2*i+1];
//     double velocity = data_copy->actuator_velocity[i];
//     double acc0 = model->actuator_acc0[i];
//     mjtNum* prmb = (mjtNum*) mju_malloc(9 * sizeof(mjtNum));
//     for (int j = 0; j<9; j++) {
//       prmb[j] = model->actuator_biasprm[10*i+j];
//     // std::cout<<"biasprm "<<j<<" "<<prmb[j]<<std::endl;
//     }
//     // std::cout<<std::endl;
//     mjtNum* prmg = (mjtNum*) mju_malloc(9 * sizeof(mjtNum));
//     for (int j = 0; j<9; j++) {
//       prmg[j] = model->actuator_gainprm[10*i+j];
//     // std::cout<<"gainprm "<<j<<" "<<prmg[j]<<std::endl;
//     }
//     // std::cout<<std::endl;

//     bias[i] = mju_muscleBias(length, lengthange, acc0, prmb);
//     gain[i] = mju_muscleGain(length, velocity, lengthange, acc0, prmg);

//     // std::cout<<"muscle "<<i<<" "<<bias[i]<<" "<<gain[i]<<std::endl;
//     // break;
//   }

//   // Eigen::MatrixXd AM = Eigen::Map<Eigen::MatrixXd>(data_copy->actuator_moment, model->nu, model->nu);

//   //matrixxd fill cols first, therefore the construction shape should be nv x nu 
//   Eigen::MatrixXd AM = Eigen::Map<Eigen::MatrixXd>(data_copy->actuator_moment, model->nv, model->nu);

//   // Eigen::SparseMatrix<double> AM_sparse = AM.sparseView();
//   // for (int i = 0; i<9; i++) {
//   //   std::cout<<"vec "<<i<<" "<<data_copy->actuator_moment[i]<<std::endl;
//   // }
//   // std::cout<<"AM "<<AM<<std::endl;
//   // std::cout<<"AM "<<AM_sparse<<std::endl;
//   // std::cout<<"AM: "<<AM(100, 0)<<" "<<AM(0, 100)<<std::endl;
//   // Eigen::MatrixXd P = 2*AM.transpose()*AM;
//   // std::cout<<"P: "<<P(100, 0)<<" "<<P(0, 100)<<std::endl;
//   mju_error_i(
//         "ctrl run end here", 0
//         );

//   // ---- ctrl computation
//   Eigen::MatrixXd P = 2 * AM.transpose() * AM;

//   // mjtNum* k_gain = (mjtNum*) mju_malloc(model->nu * sizeof(mjtNum));

  
//   Eigen::VectorXd k = AM * (gain.array() * act.array()) + AM * bias - (qfrc.array() / qfrc_scaler.array());
//   Eigen::VectorXd q = 2 * k.transpose() * AM;
//   Eigen::VectorXd lb = gain.array() * (1 - act.array()) * ts / (t2.array() + t1.array() * (1 - act.array()));
//   Eigen::VectorXd ub = - gain.array() * act.array() * ts / (t2.array() - t1.array() * act.array());
//   Eigen::VectorXd x0 = (gain.array() * (ctrl0.array() - act.array()) * ts) / ((ctrl0.array() - act.array()) * t1.array() + t2.array());
//   Eigen::VectorXd x = solve_qp(P, q, lb, ub, x0);
//   Eigen::VectorXd ctrl = act + x.array() * t2.array() / (gain.array() * ts - x.array() * t1.array());
//   for (int i = 0; i < ctrl.size(); ++i) {
//   ctrl[i] = std::clamp(ctrl[i], 0.0, 1.0);
//   }



//   double* ctrl = nullptr;


//   return ctrl;
// }

// copy policy
void HierarchicalSamplingPolicy::CopyFrom(const HierarchicalSamplingPolicy& policy, int horizon) {
  this->plan = policy.plan;
  num_spline_points = policy.num_spline_points;
}

// copy parameters
void HierarchicalSamplingPolicy::SetPlan(const TimeSpline& plan) {
  this->plan = plan;
}

}  // namespace mjpc
