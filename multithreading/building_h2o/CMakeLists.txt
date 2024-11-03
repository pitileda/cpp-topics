cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(h2o VERSION 0.1.0 LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 20)
find_package(Threads)

add_executable(h2o_barr building_h2o_barrier.cc)
target_link_libraries(h2o_barr ${CMAKE_THREADS_LIBS_INIT})

add_executable(h2o_atomic building_h2o_atomic.cc)
target_link_libraries(h2o_atomic ${CMAKE_THREADS_LIBS_INIT})

add_executable(h2o_cv building_h2o_cv.cc)
target_link_libraries(h2o_cv ${CMAKE_THREADS_LIBS_INIT})

add_executable(h2o_cv_2 building_h2o_cv_2.cc)
target_link_libraries(h2o_cv_2 ${CMAKE_THREADS_LIBS_INIT})