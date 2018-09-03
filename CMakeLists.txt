cmake_minimum_required(VERSION 3.8.0 FATAL_ERROR)
project(PlanarTrainer)
include(ExternalProject)
include (CheckIncludeFiles)
include(CheckIncludeFileCXX)

set(CMAKE_CXX_STANDARD 17)

set(USE_INSTALLED_EIGEN TRUE)
set(USE_CUSTOM_OPENCV FALSE)
set(use_theia_ransac true)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(DEPENDENCIES "${PROJECT_SOURCE_DIR}/libs")
set(LOCAL_EIGEN_REPO "https://github.com/eigenteam/eigen-git-mirror.git")
set(LOCAL_EIGEN "${DEPENDENCIES}/eigen-git-mirror")
set(LOCAL_EIGEN_GIT "${LOCAL_EIGEN}/.git")

include(CheckSymbolExists)
check_symbol_exists(localtime_r "time.h" HAVE_LOCALTIME_R)

add_compile_options(-Wno-unused-function)

set(SOURCES src/trainer.cc src/json.cc src/json.h
            src/ImageWindow.cc src/ImageWindow.hh
            src/ImageMeta.cc src/ImageMeta.h src/nanoflann.hpp
            src/CVQtScrollableImage.cc src/CVQtScrollableImage.h src/CVQtImage.cc src/CVQtImage.h
            src/Dialogs.cc src/Dialogs.h src/output.cc src/typ.h src/util.cc src/util.h
            src/hull.cc src/hull.h src/display.cc src/display.h
            src/pose/Pose.hh src/pose/pose2d.cc src/pose/pose2d.h src/pose/pose3d.cc src/pose/pose3d.h
            src/pose/poseocv.cc src/pose/poseocv.h src/pose/pose-other.h src/pose/pose-other.cc
            src/pose/PoseRANSAC.hh src/pose/PoseRANSAC.cc src/pose/math.hh src/pose/Ransac.hh
            src/pose/Optimization.cc src/pose/Optimization.h src/pose/Polynomial.cc src/pose/Polynomial.h
            src/pose/two_point_pose_partial_rotation.cc src/pose/two_point_pose_partial_rotation.h
            src/pose/five_point_relative_pose.cc src/pose/five_point_relative_pose.h
            src/ocv.cc src/ocv.h src/KeypointFlann.hh src/Calibration.hh src/pose/GravityPose.hh src/pose/OtherPose.hh)
set(INCLUDES "${PROJECT_SOURCE_DIR}/src")
set(LIBS m z)
set(FLAGS "-DQT_NO_OPENGL")

#include (CheckIncludeFiles)
#include(CheckIncludeFileCXX)
#check_include_file_cxx(filesystem HAVE_STD_FILESYSTEM)
#check_include_file_cxx(experimental/filesystem HAVE_STD_EXPERIMENTAL_FILESYSTEM)
#MESSAGE(STATUS "HAVE_STD_FILESYSTEM " ${HAVE_STD_FILESYSTEM})
#MESSAGE(STATUS "HAVE_STD_EXPERIMENTAL_FILESYSTEM " ${HAVE_STD_EXPERIMENTAL_FILESYSTEM})
#if (HAVE_STD_FILESYSTEM)
#   MESSAGE(STATUS "Using include/filesystem")
#   list(APPEND FLAGS "-DSTD_FILESYSTEM")
#   list(APPEND LIBS stdc++fs)
#elseif(HAVE_STD_EXPERIMENTAL_FILESYSTEM)
#   list(APPEND FLAGS "-DFILESYSTEM_EXPERIMENTAL")
#   list(APPEND LIBS stdc++fs)
#   MESSAGE(STATUS "Using include/experimental/filesystem")
#else()
#   FIND_PACKAGE(Boost REQUIRED COMPONENTS filesystem)
#   list(APPEND FLAGS "-DFILESYSTEM_BOOST")
#   list(APPEND INCLUDES ${Boost_INCLUDE_DIRS})
#   list(APPEND LIBS ${Boost_LIBRARIES})
#   MESSAGE(STATUS "Using filesystem from Boost")
#endif()

#Warning: Not tested in Windows
#Normally results in experimental/filesystem for C++ 14,11 and filesystem for C++ 17 at least with g++ and clang++
MESSAGE(STATUS "CMAKE_BINARY_DIR " ${CMAKE_BINARY_DIR})
file(WRITE ${CMAKE_BINARY_DIR}/filesystem_test.cc "#include <filesystem>\nint main(int argc, char** argv)\n{ std::filesystem::path p; return 0;}\n")
#try_compile is a complete abortion; it keeps trying to link with a non-existent libSTANDARD. Anyway a link is not required (assuming libstdc++fs Windows ?).
#try_compile(FILESYSTEM_COMPILE ${CMAKE_BINARY_DIR}/filesystem_test ${CMAKE_BINARY_DIR}/filesystem_test.cc LINK_LIBRARIES stdc++fs STANDARD ${CMAKE_CXX_STANDARD})
execute_process(COMMAND ${CMAKE_CXX_COMPILER} -c --std=c++${CMAKE_CXX_STANDARD} ${CMAKE_BINARY_DIR}/filesystem_test.cc
      RESULT_VARIABLE FILESYSTEM_STATUS OUTPUT_QUIET ERROR_QUIET)
MESSAGE(STATUS "C++: " ${CMAKE_CXX_STANDARD})
if (FILESYSTEM_STATUS)
   file(WRITE ${CMAKE_BINARY_DIR}/exfilesystem_test.cc "#include <experimental/filesystem>\nint main(int argc, char** argv)\n{ std::experimental::filesystem::path p; return 0;}\n")
   #try_compile(EXFILESYSTEM_COMPILE ${CMAKE_BINARY_DIR}/exfilesystem_test ${CMAKE_BINARY_DIR}/exfilesystem_test.cc LINK_LIBRARIES stdc++fs STANDARD ${CMAKE_CXX_STANDARD})
   execute_process(COMMAND ${CMAKE_CXX_COMPILER} -c --std=c++${CMAKE_CXX_STANDARD} ${CMAKE_BINARY_DIR}/exfilesystem_test.cc
         RESULT_VARIABLE EXFILESYSTEM_STATUS OUTPUT_QUIET ERROR_QUIET)
   if (EXFILESYSTEM_STATUS)
      # This might not work (without further ifdefs) as boost::filesystem does not appear to be completely compatible
      FIND_PACKAGE(Boost REQUIRED COMPONENTS filesystem)
      list(APPEND FLAGS "-DFILESYSTEM_BOOST")
      list(APPEND INCLUDES ${Boost_INCLUDE_DIRS})
      list(APPEND LIBS ${Boost_LIBRARIES})
      MESSAGE(STATUS "Using #include <boost/filesystem>")
   else()
      list(APPEND FLAGS "-DFILESYSTEM_EXPERIMENTAL")
      list(APPEND LIBS stdc++fs)
      MESSAGE(STATUS "Using #include <experimental/filesystem>")
   endif()
else()
   MESSAGE(STATUS "Using #include <filesystem>")
   list(APPEND FLAGS "-DSTD_FILESYSTEM")
   list(APPEND LIBS stdc++fs)
endif()

if (use_theia_ransac)
   list(APPEND FLAGS "-DUSE_THEIA_RANSAC")
   MESSAGE(STATUS "Using Theia RANSAC")
else()
   MESSAGE(STATUS "Using TemplRANSAC")
endif()

MESSAGE(STATUS "System: " "${CMAKE_SYSTEM}")
set(CUDA_TOOLKIT_ROOT_DIR "/opt/cuda")
set(CUSTOM_OPENCV_DIR "/opt/opencv/")
if(USE_CUSTOM_OPENCV)
#To find OpenCV at a non system prefix (system prefix /usr/lib on Linux) eg
#when the system is still on 2 and you need/want to use 3.
    set(OpenCV_FIND_QUIETLY false)
    set (OpenCV_DIR "${CUSTOM_OPENCV_DIR}")
    find_package( OpenCV REQUIRED
                  HINTS "${CMAKE_PREFIX_PATH}${CUSTOM_OPENCV_DIR}")
    if ("${OpenCV_INCLUDE_DIR}" STREQUAL "")
        set(OpenCV_INCLUDE_DIR "${CUSTOM_OPENCV_DIR}/include/")
    endif()
    list(APPEND INCLUDES "${OpenCV_INCLUDE_DIR}")
else()
#Find OpenCV at system location
    find_package( OpenCV REQUIRED)
endif()
if ("${OpenCV_INCLUDE_DIR}" STREQUAL "")
   #        set(OpenCV_INCLUDE_DIR "/usr/include/") #On Linux its empty if its in /usr/include
else()
   list(APPEND INCLUDES "${OpenCV_INCLUDE_DIR}")
endif()
list(APPEND LIBS "${OpenCV_LIBS}")

MESSAGE(STATUS "OpenCV Dir:" "${OpenCV_INSTALL_PATH}")
MESSAGE(STATUS "OpenCV Version:" "${OpenCV_VERSION_MAJOR}.${OpenCV_VERSION_MINOR}")
MESSAGE(STATUS "OpenCV include directory: " "${OpenCV_INCLUDE_DIR}")
MESSAGE(STATUS "OpenCV libs: " "${OpenCV_LIBS}")

if(USE_INSTALLED_EIGEN)
   find_package(Eigen3 REQUIRED)
else()
   file(MAKE_DIRECTORY ${LOCAL_EIGEN})
   if (EXISTS ${LOCAL_EIGEN} AND EXISTS ${LOCAL_EIGEN_GIT})
      execute_process(COMMAND git pull WORKING_DIRECTORY ${LOCAL_EIGEN})
   else()
      execute_process(COMMAND git clone ${LOCAL_EIGEN_REPO} WORKING_DIRECTORY ${DEPENDENCIES})
   endif()
   set(EIGEN3_INCLUDE_DIR "${LOCAL_EIGEN}")
endif()
MESSAGE(STATUS "Eigen 3 Include: " ${EIGEN3_INCLUDE_DIR})
list(APPEND INCLUDES "${EIGEN3_INCLUDE_DIR}")

IF (CMAKE_COMPILER_IS_GNUCXX)
   MESSAGE(STATUS "GCC Detected")
   SET(CMAKE_CXX_FLAGS "-Wall -std=c++11")
   SET(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-ggdb -O2")
   SET(CMAKE_CXX_FLAGS_RELEASE "-march=native -O2 -pipe -fstack-protector --param=ssp-buffer-size=4 -D_FORTIFY_SOURCE=2")
   #   SET(CMAKE_CXX_FLAGS_DEBUG  "-O0 -g3 -ggdb")
   SET(CMAKE_CXX_FLAGS_DEBUG  "-O0 -g3 -gdwarf-4")
   list(APPEND FLAGS "-DISGPP" )
ENDIF()

if (UNIX)
    list(APPEND FLAGS "-DLINUX")
endif (UNIX)

find_package(YamlCpp05 REQUIRED)
list(APPEND INCLUDES "${YAMLCPP_INCLUDE_DIR}")
list(APPEND LIBS "${YAMLCPP_LIBRARY}")
#if (YAMLCPP_FOUND)
#   message(STATUS "YAML include:" "${YAMLCPP_INCLUDE_DIR}")
#   message(STATUS "YAML lib:" "${YAMLCPP_LIBRARY}")
#   list(APPEND FLAGS "-DYAMLCPP")
#   list(APPEND INCLUDES "${YAMLCPP_INCLUDE_DIR}")
#   list(APPEND LIBS "${YAMLCPP_LIBRARY}")
#else()
#   message(STATUS "yaml-cpp not found - loading yaml files not available")
#endif()

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
find_package( Threads REQUIRED )
MESSAGE(STATUS "Threads Lib: " ${CMAKE_THREAD_LIBS_INIT})
MESSAGE(STATUS "Using PTHreads?: "  ${CMAKE_USE_PTHREADS_INIT})

if (CMAKE_USE_PTHREADS_INIT)
   list(APPEND FLAGS "-DPTHREADS")
endif (CMAKE_USE_PTHREADS_INIT)

list(APPEND FLAGS "-DFILESYSTEM_EXPERIMENTAL" )

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)
find_package(Qt5 COMPONENTS Core Widgets REQUIRED)
list(APPEND LIBS Qt5::Widgets)

add_library( theia-ransac theia-RANSAC/random.h theia-RANSAC/random.cc theia-RANSAC/estimator.h
             theia-RANSAC/mle_quality_measurement.h
             theia-RANSAC/random_sampler.cc theia-RANSAC/ransac.h theia-RANSAC/sampler.h theia-RANSAC/inlier_support.h
             theia-RANSAC/quality_measurement.h theia-RANSAC/random_sampler.h theia-RANSAC/sample_consensus_estimator.h
             theia-RANSAC/create_and_initialize_ransac_variant.h theia-RANSAC/prosac.h theia-RANSAC/prosac_sampler.cc
             theia-RANSAC/prosac_sampler.h theia-RANSAC/exhaustive_ransac.h theia-RANSAC/exhaustive_sampler.cc
             theia-RANSAC/exhaustive_sampler.h)
target_include_directories( theia-ransac PUBLIC ${CMAKE_SOURCE_DIR}/theia-RANSAC ${EIGEN3_INCLUDE_DIR})


add_executable(planartrainer ${SOURCES})
target_compile_options( planartrainer PRIVATE ${FLAGS} )
target_include_directories(planartrainer PUBLIC ${INCLUDES})
target_link_libraries(planartrainer theia-ransac ${LIBS})

install(TARGETS planartrainer RUNTIME DESTINATION bin)
