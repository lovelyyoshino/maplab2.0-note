cmake_minimum_required(VERSION 2.8.3)
project(loop_closure_handler)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

#############
# LIBRARIES #
#############

cs_add_library(${PROJECT_NAME}
  src/inlier-index-with-reprojection-error.cc
  src/loop-closure-handler.cc
  src/loop-detector-node.cc
  src/visualization/loop-closure-visualizer.cc)

##########
# GTESTS #
##########
catkin_add_gtest(test_loop_closure_handling_test
  test/test_loop_closure_handling_test.cc)
target_link_libraries(test_loop_closure_handling_test ${PROJECT_NAME})

##########
# EXPORT #
##########
cs_install()
cs_export()
