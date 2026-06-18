// Copyright 2021 DeepMind Technologies Limited
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

#include "mjpc/app.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <ctime>

#include <absl/flags/flag.h>
#include <absl/random/random.h>
#include <mujoco/mujoco.h>
#include <glfw_adapter.h>
#include <Eigen/Dense>
#include "mjpc/array_safety.h"
#include "mjpc/agent.h"
#include "mjpc/estimators/estimator.h"
#include "mjpc/simulate.h"  // mjpc fork
#include "mjpc/task.h"
#include "mjpc/threadpool.h"
#include "mjpc/utilities.h"

//TODO: changed default setting
ABSL_FLAG(bool, planner_enabled, true,
          "If true, the planner will run on startup");
ABSL_FLAG(float, sim_percent_realtime, 1,
          "The realtime percentage at which the simulation will be launched.");// recommend: walk: 2.5, hand: 1
ABSL_FLAG(bool, estimator_enabled, false,
          "If true, estimator loop will run on startup");
ABSL_FLAG(bool, show_left_ui, true,
          "If true, the left UI (ui0) will be visible on startup");
ABSL_FLAG(bool, show_plot, true,
          "If true, the plots will be visible on startup");
ABSL_FLAG(bool, show_info, true,
          "If true, the infotext panel will be visible on startup");


namespace {
namespace mj = ::mujoco;
namespace mju = ::mujoco::util_mjpc;

// maximum mis-alignment before re-sync (simulation seconds)
const double syncMisalign = 0.1;

// fraction of refresh available for simulation
const double simRefreshFraction = 0.7;

// model and data
mjModel* m = nullptr;
mjData* d = nullptr;

// control noise variables
mjtNum* ctrlnoise = nullptr;

// temp file to store cost
// std::string tmp_filename = "cost.txt";
std::ofstream output_file;

using Seconds = std::chrono::duration<double>;

// --------------------------------- callbacks ---------------------------------
std::unique_ptr<mj::Simulate> sim;



// controller
extern "C" {
void controller(const mjModel* m, mjData* d);
}

absl::BitGen gen_;
double rand_dir[3] = {
                        absl::Gaussian<double>(gen_, 0.0, 1), 
                        absl::Gaussian<double>(gen_, 0.0, 1),
                        absl::Gaussian<double>(gen_, 0.0, 1)
                        };

// controller callback
void controller(const mjModel* m, mjData* data) {
  // if agent, skip
  if (data != d) {
    return;
  }

  // if simulation:
  if (sim->agent->action_enabled) {
    sim->agent->ActivePlanner().ActionFromPolicy(
        data->ctrl, &sim->agent->state.state()[0],
        sim->agent->state.time());
    if (m->nv >=100) {
      sim->agent->ActivePlanner().SyncTargetPos(data);
    }
  }
  // if noise
  if (!sim->agent->allocate_enabled && sim->uiloadrequest.load() == 0 &&
      sim->ctrl_noise_std) {
    for (int j = 0; j < sim->m->nu; j++) {
      data->ctrl[j] += ctrlnoise[j];
    }
  }

  // for track, care about the avg pos
  if (sim->agent->ActiveTask()->Name() == "MS Track5" 
  || sim->agent->ActiveTask()->Name() == "MS Track4" 
  || sim->agent->ActiveTask()->Name() == "MS Track3" 
  || sim->agent->ActiveTask()->Name() == "MS Track2"
  ) {
    double collect_time = 3;
    double single_cost_sum = 0;
    for (int i = 0; i < 16; i++) {
      single_cost_sum += mju_norm3(data->sensordata+782+3*i) * mju_norm3(data->sensordata+782+3*i); // avg square
    }
    sim->cost_sum += single_cost_sum;

    if (data->time > collect_time 
       || sim->previous_time_step > data->time || mju_abs(data->qpos[3]) > 1 || mju_abs(data->qpos[4]) > 1
    ) {
      if (data->time < collect_time) {
        // double current_step = sim->previous_time_step/m->opt.timestep;
        double lost_step = (collect_time - sim->previous_time_step)/m->opt.timestep;
        sim->cost_sum += lost_step * 5;
        std::cout<<"diverge early stop "<<" "<<lost_step<<" "<<sim->cost_sum<<std::endl;
      }
      output_file.open("cost.txt", std::ios::out);
      output_file << sim->cost_sum <<'\n';
      output_file << sim->qpos_str;
      std::cout<<"store"<<std::endl;
      output_file.close();
      sim->exitrequest.store(1);
    
    }

    sim->previous_time_step = data->time;
    for (int i = 0; i < m->nq; i++) {
      sim->qpos_str += std::to_string(data->qpos[i]);
      if (i < m->nq - 1) {
        sim->qpos_str += ",";
      }
    }
    sim->qpos_str += "\n";
  }

  else if (sim->agent->ActiveTask()->Name() == "MS HandCube" || 
           sim->agent->ActiveTask()->Name() == "MS HandBottle") {    
    for (int i = 0; i < m->nq; i++) {
      sim->qpos_str += std::to_string(data->qpos[i]);
      if (i < m->nq - 1) {
        sim->qpos_str += ",";
      }
    }
    sim->qpos_str += "\n";

    double single_cost_sum = 0;
    for (int i = 0; i < 16; i++) {
      single_cost_sum += mju_norm3(data->sensordata+782+3*i) * mju_norm3(data->sensordata+782+3*i); // avg square
    }
    sim->cost_sum += single_cost_sum;

    // Stop after 5 seconds, shorter may be too short for vlm input
    if (data->time >= 5) {
      const int MAX_LEN = 80;
      char s[MAX_LEN];
      time_t t = time(0);
      strftime(s, MAX_LEN, "%Y_%m_%d_%H_%M_%S", localtime(&t));
      std::string time_str = s;

      std::string qpos_file = time_str + "_qpos.txt";

      output_file.open(qpos_file, std::ios::out);
      output_file << sim->qpos_str;
      output_file.close();
      sim->exitrequest.store(1);

      output_file.open("cost.txt", std::ios::out);
      output_file << sim->cost_sum <<'\n';
      output_file << sim->qpos_str;
      std::cout<<"store"<<std::endl;
      output_file.close();

    }
  }

  else if (sim->agent->ActiveTask()->Name() == "MS WalkGym2" ||
           sim->agent->ActiveTask()->Name() == "MS Rough" ||
           sim->agent->ActiveTask()->Name() == "MS Slope" ||
           sim->agent->ActiveTask()->Name() == "MS Stair" || 
           sim->agent->ActiveTask()->Name() == "MS WalkOstrich") {
    double* pelvis_position = mjpc::SensorByName(m, data, "pelvis_position");

    // qpos data
    for (int i = 0; i < m->nq; i++) {
      sim->qpos_str += std::to_string(data->qpos[i]);
      if (i < m->nq - 1) {
        sim->qpos_str += ",";
      }
    }
    sim->qpos_str += "\n";

    // Muscle Activation data
    for (int i = 0; i < m->nu; i++) {
      sim->act_str += std::to_string(data->act[i]);
      if (i < m->nu - 1) {
        sim->act_str += ",";
      }
    }
    sim->act_str += "\n";

    if (data->time >= 10 || (pelvis_position[2] < 0.4 && data->time >= 3)) {
      std::cout << "saving file" << std::endl;
      const int MAX_LEN = 80;
      char s[MAX_LEN];
      time_t t = time(0);
      strftime(s, MAX_LEN, "%Y_%m_%d_%H_%M_%S", localtime(&t));
      std::string time_str = s;

      std::string qpos_file = time_str + "_qpos.txt";
      std::cout << "qpos_file: " << qpos_file << std::endl;

      output_file.open(qpos_file, std::ios::out);
      output_file << sim->qpos_str;
      output_file.close();

      std::string act_file = time_str+"_act.txt";
      output_file.open(act_file, std::ios::out);
      output_file << sim->act_str;
      output_file.close();

      sim->exitrequest.store(1);
    }
  }
  
  else {
    // qpos
    for (int i = 0; i < m->nq; i++) {
      sim->qpos_str += std::to_string(data->qpos[i]);
      if (i < m->nq - 1) {
        sim->qpos_str += ",";
      }
    }
    sim->qpos_str += "\n";

    // act
    for (int i = 0; i < m->nu; i++) {
      sim->act_str += std::to_string(data->act[i]);
      if (i < m->nu - 1) {
        sim->act_str += ",";
      }
    }
    sim->act_str += "\n";

    //actuator force
    for (int i = 0; i < m->nu; i++) {
      sim->actuator_force_str += std::to_string(data->actuator_force[i]);
      if (i < m->nu - 1) {
        sim->actuator_force_str += ",";
      }
    }
    sim->actuator_force_str += "\n";

    //actuator torque
    for (int i = 0; i < m->nv; i++) {
      sim->actuator_torque_str += std::to_string(data->qfrc_actuator[i]);
      if (i < m->nv - 1) {
        sim->actuator_torque_str += ",";
      }
    }
    sim->actuator_torque_str += "\n";

    if (sim->previous_time_step > data->time || data->time > 30
    //  or com_position[0]>=10
     ) {
      // output_file << sim->cost_sum <<'\n';
      const int MAXLEN = 800;
      char s[MAXLEN];
      // std::cout<<"store1"<<std::endl;
      time_t t = time(0);
      strftime(s, MAXLEN, "%Y_%m_%d_%H_%M_%S", localtime(&t));
      std::string time_str = s;
      time_str += "_" + sim->agent->ActiveTask()->Name();
      //Remove space
      time_str.erase(std::remove(time_str.begin(), time_str.end(), ' '), time_str.end());

      std::string qpos_file = time_str+"_qpos.txt";
      std::string act_file = time_str+"_act.txt";
      // std::string xfrc_file = time_str+"_xfrc.txt";
      // std::string com_file = time_str+"_com.txt";
      std::string actuator_force_file = time_str+"_actuator_force.txt";
      std::string actuator_torque_file = time_str+"_actuator_torque.txt";
      // std::string actuator_moment_file = time_str+"_actuator_moment.txt";

      // output_file.open("cost.txt", std::ios::out);
      // output_file << std::to_string(com_position[0]/data->time) <<'\n';
      // output_file.close();

      output_file.open(qpos_file, std::ios::out);
      output_file << sim->qpos_str;
      output_file.close();

      output_file.open(act_file, std::ios::out);
      output_file << sim->act_str;
      output_file.close();

      output_file.open(actuator_force_file, std::ios::out);
      output_file << sim->actuator_force_str;
      output_file.close();

      output_file.open(actuator_torque_file, std::ios::out);
      output_file << sim->actuator_torque_str;
      output_file.close();
      
      std::cout<<"store"<<std::endl;
      
      sim->exitrequest.store(1);
    }
    sim->previous_time_step = data->time; 
  }
}

// sensor
extern "C" {
void sensor(const mjModel* m, mjData* d, int stage);
}

// sensor callback
void sensor(const mjModel* model, mjData* data, int stage) {
  if (stage == mjSTAGE_ACC) {
    if (!sim->agent->allocate_enabled && sim->uiloadrequest.load() == 0) {
      
      if (sim->agent->IsPlanningModel(model)) {
        // the planning thread and rollout threads don't need
        // synchronization when using PlanningResidual.
        const mjpc::ResidualFn* residual = sim->agent->PlanningResidual();
        residual->Residual(model, data, data->sensordata);
        
      } else {
        // this residual is used by the physics thread and the UI thread (for
        // plots), and is run with a shared lock, to safely run with changes to
        // weights and parameters
        sim->agent->ActiveTask()->Residual(model, data, data->sensordata);
      }
      
    }
  }
}

//--------------------------------- simulation ---------------------------------

mjModel* LoadModel(const mjpc::Agent* agent, mj::Simulate& sim) {
  mjpc::Agent::LoadModelResult load_model = sim.agent->LoadModel();
  mjModel* mnew = load_model.model.release();
  mju::strcpy_arr(sim.load_error, load_model.error.c_str());

  if (!mnew) {
    std::cout << load_model.error << "\n";
    return nullptr;
  }

  // compiler warning: print and pause
  if (!load_model.error.empty()) {
    std::cout << "Model compiled, but simulation warning (paused):\n  "
              << load_model.error << "\n";
    sim.run = 0;
  }

  return mnew;
}

// estimator in background thread
void EstimatorLoop(mj::Simulate& sim) {
  // run until asked to exit
  while (!sim.exitrequest.load()) {
    if (sim.uiloadrequest.load() == 0) {
      // estimator
      int active_estimator = sim.agent->ActiveEstimatorIndex();
      mjpc::Estimator* estimator = &sim.agent->ActiveEstimator();

      // estimator update
      if (!active_estimator) {
        std::this_thread::yield();
        continue;
      } else {
        // start timer
        auto start = std::chrono::steady_clock::now();

        // set values from GUI
        estimator->SetGUIData();

        // get simulation state (lock physics thread)
        {
          const std::lock_guard<std::mutex> lock(sim.mtx);
          // copy simulation ctrl
          mju_copy(sim.agent->ctrl.data(), d->ctrl, m->nu);

          // copy simulation sensor
          mju_copy(sim.agent->sensor.data(), d->sensordata, m->nsensordata);

          // copy simulation time
          estimator->Data()->time = d->time;

          // copy simulation mocap
          mju_copy(estimator->Data()->mocap_pos, d->mocap_pos, 3 * m->nmocap);
          mju_copy(estimator->Data()->mocap_quat, d->mocap_quat, 4 * m->nmocap);

          // copy simulation userdata
          mju_copy(estimator->Data()->userdata, d->userdata, m->nuserdata);
        }

        // update filter using latest ctrl and sensor copied from physics thread
        estimator->Update(sim.agent->ctrl.data(), sim.agent->sensor.data());

        // estimator state to planner
        double* state = estimator->State();
        sim.agent->state.Set(m, state, state + m->nq, state + m->nq + m->nv,
                             d->mocap_pos, d->mocap_quat, d->userdata, d->time, 
                             d->actuator_length, d->actuator_velocity,
                             d->xfrc_applied);

        // wait (us)
        // TODO(taylor): confirm valid for slowdown
        while (mjpc::GetDuration(start) <
               1.0e6 * estimator->Model()->opt.timestep) {
        }
      }
    }
  }
}

// simulate in background thread (while rendering in main thread)
void PhysicsLoop(mj::Simulate& sim) {
  // cpu-sim synchronization point
  std::chrono::time_point<mj::Simulate::Clock> syncCPU;
  mjtNum syncSim = 0;

  // run until asked to exit
  while (!sim.exitrequest.load()) {
    if (sim.droploadrequest.load()) {
      // TODO(nimrod): Implement drag and drop support in MJPC
    }

    // ----- task reload ----- //
    if (sim.uiloadrequest.load() == 1) {
      // get new model + task
      sim.filename = sim.agent->GetTaskXmlPath(sim.agent->gui_task_id);

      mjModel* mnew = LoadModel(sim.agent.get(), sim);
      mjData* dnew = nullptr;
      if (mnew) dnew = mj_makeData(mnew);
      if (dnew) {
        sim.agent->Initialize(mnew);
        sim.agent->plot_enabled = absl::GetFlag(FLAGS_show_plot);
        sim.agent->plan_enabled = absl::GetFlag(FLAGS_planner_enabled);
        sim.agent->Allocate();

        // set home keyframe
        int home_id = mj_name2id(mnew, mjOBJ_KEY, "home");
        if (home_id >= 0) {
          mj_resetDataKeyframe(mnew, dnew, home_id);
          sim.agent->Reset(dnew->ctrl);
        } else {
          sim.agent->Reset();
        }
        sim.agent->PlotInitialize();

        sim.Load(mnew, dnew, sim.filename, true);
        m = mnew;
        d = dnew;
        mj_forward(m, d);

        // allocate ctrlnoise
        free(ctrlnoise);
        ctrlnoise = static_cast<mjtNum*>(malloc(sizeof(mjtNum) * m->nu));
        mju_zero(ctrlnoise, m->nu);
      }

      // decrement counter
      sim.uiloadrequest.fetch_sub(1);
    }

    // reload GUI
    if (sim.uiloadrequest.load() == -1) {
      sim.Load(sim.m, sim.d, sim.filename.c_str(), false);
      sim.uiloadrequest.fetch_add(1);
    }
    // ----------------------- //

    // sleep for 1 ms or yield, to let main thread run
    //  yield results in busy wait - which has better timing but kills battery
    //  life
    if (sim.run && sim.busywait) {
      std::this_thread::yield();
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    {
      // lock the sim mutex
      const std::lock_guard<std::mutex> lock(sim.mtx);

      if (m) {  // run only if model is present
        sim.agent->ActiveTask()->Transition(m, d);

        // running
        if (sim.run) {
          // record cpu time at start of iteration
          const auto startCPU = mj::Simulate::Clock::now();

          // elapsed CPU and simulation time since last sync
          const auto elapsedCPU = startCPU - syncCPU;
          double elapsedSim = d->time - syncSim;

          // inject noise
          if (sim.ctrl_noise_std) {
            // convert rate and scale to discrete time (Ornstein–Uhlenbeck)
            mjtNum rate = mju_exp(-m->opt.timestep / sim.ctrl_noise_rate);
            mjtNum scale = sim.ctrl_noise_std * mju_sqrt(1 - rate * rate);

            for (int i = 0; i < m->nu; i++) {
              // update noise
              ctrlnoise[i] =
                  rate * ctrlnoise[i] + scale * mju_standardNormal(nullptr);
            }
          }

          // requested slow-down factor
          double slowdown = 100 / sim.percentRealTime[sim.real_time_index];

          // misalignment condition: distance from target sim time is bigger
          // than maximum misalignment `syncMisalign`
          bool misaligned = mju_abs(Seconds(elapsedCPU).count() / slowdown -
                                    elapsedSim) > syncMisalign;

          // out-of-sync (for any reason): reset sync times, step
          if (elapsedSim < 0 || elapsedCPU.count() < 0 ||
              syncCPU.time_since_epoch().count() == 0 || misaligned ||
              sim.speed_changed) {
            // re-sync
            syncCPU = startCPU;
            syncSim = d->time;
            sim.speed_changed = false;

            // clear old perturbations, apply new
            mju_zero(d->xfrc_applied, 6 * m->nbody);
            sim.ApplyPosePerturbations(0);  // move mocap bodies only
            sim.ApplyForcePerturbations();

            // run single step, let next iteration deal with timing
            sim.agent->ExecuteAllRunBeforeStepJobs(m, d);
            mj_step(m, d);
          } else {  // in-sync: step until ahead of cpu
            bool measured = false;
            mjtNum prevSim = d->time;
            double refreshTime = simRefreshFraction / sim.refresh_rate;

            // step while sim lags behind cpu and within refreshTime
            while (Seconds((d->time - syncSim) * slowdown) <
                       mj::Simulate::Clock::now() - syncCPU &&
                   mj::Simulate::Clock::now() - startCPU <
                       Seconds(refreshTime)) {
              // measure slowdown before first step
              if (!measured && elapsedSim) {
                sim.measured_slowdown =
                    std::chrono::duration<double>(elapsedCPU).count() /
                    elapsedSim;
                measured = true;
              }

              // clear old perturbations, apply new
              mju_zero(d->xfrc_applied, 6 * m->nbody);
              sim.ApplyPosePerturbations(0);  // move mocap bodies only
              sim.ApplyForcePerturbations();

              // call mj_step
              sim.agent->ExecuteAllRunBeforeStepJobs(m, d);
              mj_step(m, d);

              // break if reset
              if (d->time < prevSim) {
                break;
              }
            }
          }
        } else {  // paused
          // apply pose perturbation
          sim.ApplyPosePerturbations(1);  // move mocap and dynamic bodies

          // still accept jobs when simulation is paused
          sim.agent->ExecuteAllRunBeforeStepJobs(m, d);

          // run mj_forward, to update rendering and joint sliders
          mj_forward(m, d);
          sim.speed_changed = true;
        }
      }
    }  // release sim.mtx

    // state
    if (sim.uiloadrequest.load() == 0) {
      // set ground truth state if no active estimator
      if (!sim.agent->ActiveEstimatorIndex() || !sim.agent->estimator_enabled) {
        sim.agent->state.Set(m, d);
      }
    }
  }
}
}  // namespace

// ------------------------------- main ----------------------------------------

namespace mjpc {

MjpcApp::MjpcApp(std::vector<std::shared_ptr<mjpc::Task>> tasks, int task_id) {
  // MJPC
  printf("MuJoCo MPC (MJPC)\n");

  // MuJoCo
  std::printf(" MuJoCo version %s\n", mj_versionString());
  if (mjVERSION_HEADER != mj_version()) {
    mju_error("Headers and library have Different versions");
  }

  // threads
  printf(" Hardware threads:  %i\n", mjpc::NumAvailableHardwareThreads());

  if (sim != nullptr) {
    mju_error("Multiple instances of MjpcApp created.");
    return;
  }
  sim = std::make_unique<mj::Simulate>(
      std::make_unique<mujoco::GlfwAdapter>(),
      std::make_shared<Agent>());

  sim->agent->SetTaskList(std::move(tasks));
  sim->agent->gui_task_id = task_id;

  sim->filename = sim->agent->GetTaskXmlPath(sim->agent->gui_task_id);
  printf("before load model\n");
  m = LoadModel(sim->agent.get(), *sim);
  printf("after load model\n");

  if (m) d = mj_makeData(m);

  // set home keyframe
  int home_id = mj_name2id(m, mjOBJ_KEY, "home");
  if (home_id >= 0) mj_resetDataKeyframe(m, d, home_id);

  sim->mnew = m;
  sim->dnew = d;

  // control noise
  free(ctrlnoise);
  ctrlnoise = (mjtNum*)malloc(sizeof(mjtNum) * m->nu);
  mju_zero(ctrlnoise, m->nu);
  printf("before agent\n");
  // agent
  sim->agent->estimator_enabled = absl::GetFlag(FLAGS_estimator_enabled);
  sim->agent->Initialize(m);
  printf("agent init\n");
  sim->agent->Allocate();
  printf("agent allocate\n");
  sim->agent->Reset();
  printf("agent reset\n");
  sim->agent->PlotInitialize();
  printf("agent plot\n");

  sim->agent->plan_enabled = absl::GetFlag(FLAGS_planner_enabled);
  printf("after agent\n");
  // Get the index of the closest sim percentage to the input.
  float desired_percent = absl::GetFlag(FLAGS_sim_percent_realtime);
  sim->desired_percent_real_time = desired_percent;
  std::cout<<"desired percent "<<desired_percent<<std::endl;
  auto closest = std::min_element(
      std::begin(sim->percentRealTime), std::end(sim->percentRealTime),
      [&](float a, float b) {
        return std::abs(a - desired_percent) < std::abs(b - desired_percent);
      });
  sim->real_time_index =
      std::distance(std::begin(sim->percentRealTime), closest);
  std::cout<<"real time index "<<sim->real_time_index<<std::endl;
  sim->delete_old_m_d = true;
  sim->loadrequest = 2;

  sim->ui0_enable = absl::GetFlag(FLAGS_show_left_ui);
  sim->info = absl::GetFlag(FLAGS_show_info);
}

MjpcApp::~MjpcApp() {
  sim.reset();
}

// run event loop
void MjpcApp::Start() {
  // threads
  printf("  physics        :  %i\n", 1);
  printf("  render         :  %i\n", 1);
  printf("  Planner        :  %i\n", 1);
  #include <fstream> // Include the necessary header file for file stream operations

  printf("    planning     :  %i\n", sim->agent->planner_threads());
  printf("  Estimator      :  %i\n", sim->agent->estimator_threads());
  printf("    estimation   :  %i\n", sim->agent->estimator_enabled);

  // set control callback
  mjcb_control = controller;
  printf("start");
  // set sensor callback
  mjcb_sensor = sensor;

  // one-off preparation:
  sim->InitializeRenderLoop();

  // start physics thread
  mjpc::ThreadPool physics_pool(1);
  physics_pool.Schedule([]() { PhysicsLoop(*sim); });

  // start estimator thread
  mjpc::ThreadPool estimator_pool(1);
  if (sim->agent->estimator_enabled) {
    estimator_pool.Schedule([]() { EstimatorLoop(*sim); });
  }

  {
    // start plan thread
    mjpc::ThreadPool plan_pool(1);
    plan_pool.Schedule(
        []() { sim->agent->Plan(sim->exitrequest, sim->uiloadrequest); });
    // start simulation UI loop (blocking call)
    sim->RenderLoop();
  }
}

mj::Simulate* MjpcApp::Sim() {
  return sim.get();
}

void StartApp(std::vector<std::shared_ptr<mjpc::Task>> tasks, int task_id) {
  printf("into start app %i\n", task_id);
  MjpcApp app(std::move(tasks), task_id);
  printf("init app\n");
  app.Start();
}

}  // namespace mjpc
