cmake_minimum_required (VERSION 2.8)
project(loop_closure_plugin)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

add_definitions(-fPIC -shared)

cs_add_library(${PROJECT_NAME} src/loop-closure-plugin.cc
                               src/vi-map-merger.cc)
create_console_plugin(${PROJECT_NAME})

##################
# CMAKE Indexing #
##################
FILE(GLOB_RECURSE LibFiles "include/*")
add_custom_target(headers SOURCES ${LibFiles})

cs_install()
cs_export()
