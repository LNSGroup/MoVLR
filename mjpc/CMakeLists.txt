# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set(CMAKE_LIBRARY_PATH "/home/anaconda3/envs/lib")
set(CMAKE_INCLUDE_PATH "/home/anaconda3/envs/include")
set(CMAKE_LIBRARY_PATH "/usr/local/lib")
set(CMAKE_INCLUDE_PATH "/usr/local/include")
# set(CMAKE_LIBRARY_PATH "/home/anaconda3/lib")
# set(CMAKE_INCLUDE_PATH "/home/anaconda3/include")
find_package(OsqpEigen REQUIRED)

# find_package(libglpk REQUIRED)
# add_executable(example example.cpp)
# target_link_libraries(
#   OsqpEigen::OsqpEigen
#   )


add_library(threadpool STATIC)
target_sources(
  threadpool
  PUBLIC threadpool.h
  PRIVATE threadpool.cc
)
target_link_libraries(
  threadpool
  absl::base
  
)
target_include_directories(threadpool PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

add_library(
  libmjpc STATIC
  states/state.cc
  states/state.h
  agent.cc
  agent.h
  trajectory.cc
  trajectory.h
  utilities.cc
  utilities.h
  tasks/tasks.cc
  tasks/tasks.h
  tasks/acrobot/acrobot.cc
  tasks/acrobot/acrobot.h
  tasks/allegro/allegro.cc
  tasks/allegro/allegro.h
  tasks/bimanual/insert/insert.cc
  tasks/bimanual/insert/insert.h
  tasks/bimanual/handover/handover.cc
  tasks/bimanual/handover/handover.h
  tasks/bimanual/reorient/reorient.cc
  tasks/bimanual/reorient/reorient.h
  tasks/cartpole/cartpole.cc
  tasks/cartpole/cartpole.h
  tasks/fingers/fingers.cc
  tasks/fingers/fingers.h
  tasks/humanoid/interact/interact.cc
  tasks/humanoid/interact/interact.h
  tasks/humanoid/interact/contact_keyframe.cc
  tasks/humanoid/interact/contact_keyframe.h
  tasks/humanoid/stand/stand.cc
  tasks/humanoid/stand/stand.h
  tasks/humanoid/tracking/tracking.cc
  tasks/humanoid/tracking/tracking.h
  tasks/humanoid/walk/walk.cc
  tasks/humanoid/walk/walk.h
  tasks/manipulation/common.cc
  tasks/manipulation/common.h
  tasks/manipulation/manipulation.cc
  tasks/manipulation/manipulation.h
  tasks/musculoskeletal/musculoskeletal.cc
  tasks/musculoskeletal/musculoskeletal.h
  tasks/musculoskeletal/stand/stand.cc
  tasks/musculoskeletal/stand/stand.h
  tasks/musculoskeletal/walk/walk.cc
  tasks/musculoskeletal/walk/walk.h
  tasks/musculoskeletal/walk2/walk.cc
  tasks/musculoskeletal/walk2/walk.h
  tasks/musculoskeletal/walk3/walk.cc
  tasks/musculoskeletal/walk3/walk.h
  tasks/musculoskeletal/walk4/walk.cc
  tasks/musculoskeletal/walk4/walk.h
  tasks/musculoskeletal/stair/walk.cc
  tasks/musculoskeletal/stair/walk.h
  tasks/musculoskeletal/slope/walk.cc
  tasks/musculoskeletal/slope/walk.h
  tasks/musculoskeletal/rough/walk.cc
  tasks/musculoskeletal/rough/walk.h
  tasks/musculoskeletal/tracking/tracking.cc
  tasks/musculoskeletal/tracking/tracking.h
  tasks/musculoskeletal/tracking2/tracking.cc
  tasks/musculoskeletal/tracking2/tracking.h
  tasks/musculoskeletal/tracking3/tracking.cc
  tasks/musculoskeletal/tracking3/tracking.h
  tasks/musculoskeletal/tracking4/tracking.cc
  tasks/musculoskeletal/tracking4/tracking.h
  tasks/musculoskeletal/tracking5/tracking.cc
  tasks/musculoskeletal/tracking5/tracking.h
  tasks/musculoskeletal/walk_gym/walk.cc
  tasks/musculoskeletal/walk_gym/walk.h
  tasks/musculoskeletal/walk_gym2/walk.cc
  tasks/musculoskeletal/walk_gym2/walk.h
  tasks/musculoskeletal/run_gym/walk.cc
  tasks/musculoskeletal/run_gym/walk.h
  tasks/musculoskeletal/walk_ostrich/walk.cc
  tasks/musculoskeletal/walk_ostrich/walk.h
  tasks/musculoskeletal/walk_mice/walk.cc
  tasks/musculoskeletal/walk_mice/walk.h
  tasks/musculoskeletal/stand_gym/walk.cc
  tasks/musculoskeletal/stand_gym/walk.h
  tasks/musculoskeletal/hand_cube/hand.cc
  tasks/musculoskeletal/hand_cube/hand.h
  tasks/musculoskeletal/hand_bottle/hand.cc
  tasks/musculoskeletal/hand_bottle/hand.h
  tasks/musculoskeletal/swim/walk.cc
  tasks/musculoskeletal/swim/walk.h
  tasks/op3/stand.cc
  tasks/op3/stand.h
  tasks/panda/panda.cc
  tasks/panda/panda.h
  tasks/particle/particle.cc
  tasks/particle/particle.h
  tasks/rubik/solve.cc
  tasks/rubik/solve.h
  tasks/shadow_reorient/hand.cc
  tasks/shadow_reorient/hand.h
  tasks/quadrotor/quadrotor.cc
  tasks/quadrotor/quadrotor.h
  tasks/quadruped/quadruped.cc
  tasks/quadruped/quadruped.h
  tasks/swimmer/swimmer.cc
  tasks/swimmer/swimmer.h
  tasks/walker/walker.cc
  tasks/walker/walker.h
  planners/planner.cc
  planners/planner.h
  planners/policy.h
  planners/include.cc
  planners/include.h
  # planners/cost_derivatives.cc
  # planners/cost_derivatives.h
  # planners/model_derivatives.cc
  # planners/model_derivatives.h
  planners/cross_entropy/planner.cc
  planners/cross_entropy/planner.h
  planners/hierarchical_cross_entropy/planner.cc
  planners/hierarchical_cross_entropy/planner.h
  # planners/robust/robust_planner.cc
  # planners/robust/robust_planner.h
  # planners/sample_gradient/planner.cc
  # planners/sample_gradient/planner.h
  planners/sampling/planner.cc
  planners/sampling/planner.h
  planners/sampling/policy.cc
  planners/sampling/policy.h
  # planners/hierarchical_sampling/planner.cc
  # planners/hierarchical_sampling/planner.h
  # planners/hierarchical_sampling/policy.cc
  # planners/hierarchical_sampling/policy.h
  planners/hierarchical_pd/planner.cc
  planners/hierarchical_pd/planner.h
  planners/hierarchical_pd/policy.cc
  planners/hierarchical_pd/policy.h
  planners/hierarchical_mppi/planner.cc
  planners/hierarchical_mppi/planner.h
  # planners/gradient/gradient.cc
  # planners/gradient/gradient.h
  # planners/gradient/planner.cc
  # planners/gradient/planner.h
  # planners/gradient/policy.cc
  # planners/gradient/policy.h
  # planners/gradient/settings.h
  # planners/gradient/spline_mapping.cc
  # planners/gradient/spline_mapping.h
  # planners/ilqg/backward_pass.cc
  # planners/ilqg/backward_pass.h
  # planners/ilqg/boxqp.h
  # planners/ilqg/planner.cc
  # planners/ilqg/planner.h
  # planners/ilqg/policy.cc
  # planners/ilqg/policy.h
  # planners/ilqg/settings.h
  # planners/ilqs/planner.cc
  # planners/ilqs/planner.h
  estimators/batch.cc
  estimators/batch.h
  estimators/estimator.h
  estimators/include.cc
  estimators/include.h
  estimators/kalman.cc
  estimators/kalman.h
  estimators/unscented.cc
  estimators/unscented.h
  direct/direct.cc
  direct/direct.h
  direct/trajectory.h
  direct/model_parameters.cc
  direct/model_parameters.h
  spline/spline.cc
  spline/spline.h
  app.cc
  app.h
  norm.cc
  norm.h
  simulate.cc
  simulate.h
  task.cc
  task.h
  $<TARGET_OBJECTS:mujoco::platform_ui_adapter>
)
set_target_properties(libmjpc PROPERTIES OUTPUT_NAME mjpc)
target_compile_options(libmjpc PUBLIC ${MJPC_COMPILE_OPTIONS})
target_compile_definitions(libmjpc PRIVATE MJSIMULATE_STATIC)
target_link_libraries(
  libmjpc
  absl::any_invocable
  absl::check
  absl::flat_hash_map
  absl::log
  absl::random_random
  absl::span
  glfw
  lodepng
  mujoco::mujoco
  mujoco::platform_ui_adapter
  threadpool
  Threads::Threads
  OsqpEigen::OsqpEigen
  # libglpk.so.40
  
)
target_include_directories(libmjpc
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${CMAKE_CURRENT_BINARY_DIR}/..
)

add_executable(
  mjpc
  main.cc
)
target_link_libraries(
  mjpc
  absl::flags
  absl::flags_parse
  absl::random_random
  absl::strings
  libmjpc
  mujoco::mujoco
  threadpool
  Threads::Threads
  OsqpEigen::OsqpEigen
  # libglpk.so.40
  
)


target_include_directories(mjpc PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_compile_options(mjpc PUBLIC ${MJPC_COMPILE_OPTIONS})
target_link_options(mjpc PRIVATE ${MJPC_LINK_OPTIONS})
target_compile_definitions(mjpc PRIVATE MJSIMULATE_STATIC)
if(APPLE)
  target_sources(mjpc PRIVATE macos_gui.mm)
  target_link_libraries(mjpc "-framework Cocoa")
endif()

add_library(
  libtestspeed STATIC
  testspeed_app.cc
  testspeed.h
  testspeed.cc
)
target_link_libraries(
  libtestspeed
  absl::random_random
  absl::strings
  libmjpc
  mujoco::mujoco
  threadpool
  Threads::Threads
  OsqpEigen::OsqpEigen
  # libglpk.so.40
  
)
target_include_directories(libtestspeed PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

add_executable(
  testspeed
  testspeed_app.cc
)
target_link_libraries(
  testspeed
  libtestspeed
  absl::flags
  absl::flags_parse
  OsqpEigen::OsqpEigen
  # libglpk.so.40
  
)
target_include_directories(testspeed PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_compile_options(testspeed PUBLIC ${MJPC_COMPILE_OPTIONS})
target_link_options(testspeed PRIVATE ${MJPC_LINK_OPTIONS})
target_compile_definitions(testspeed PRIVATE MJSIMULATE_STATIC)

add_subdirectory(tasks)

if(BUILD_TESTING AND MJPC_BUILD_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()

if(MJPC_BUILD_GRPC_SERVICE)
  add_subdirectory(grpc)
endif()



