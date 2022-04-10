cmake_minimum_required(VERSION 2.8.4)

PROJECT(EmotionGeneration)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp -std=c++11")
find_package(OpenCV REQUIRED)
find_package(assimp REQUIRED)
find_package(OpenGL)
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)
find_package(glfw3 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package (PkgConfig REQUIRED)
pkg_check_modules (GTK3 REQUIRED gtk+-3.0)
pkg_check_modules (GLIBMM REQUIRED glibmm-2.4)
pkg_check_modules (GTKMM REQUIRED gtkmm-3.0)

#SET(NNLS_SOURCES nnls.h)

add_definitions (${GTK3_CFLAGS_OTHER} ${GLIBMM_CFLAGS} ${GTKMM_CFLAGS})
include_directories(${OPENGL_INCLUDE_DIRS} ${GLEW_INCLUDE_DIRS} ${GTK3_INCLUDE_DIRS} ${GLIBMM_INCLUDE_DIRS} ${GTKMM_INCLUDE_DIRS})
link_directories (${GTK3_LIBRARY_DIRS}  ${GLIBMM_LIBRARY_DIRS} ${GTKMM_LIBRARY_DIRS})
add_executable(run_tool main.cpp data.cpp utility.cpp common/shader.cpp common/objloader.cpp common/controls.cpp common/text2D.cpp common/texture.cpp box_multiwidget.cpp CorrectiveWeightsAnalytic.cpp)
target_link_libraries(run_tool glfw EGL ${OpenCV_LIBS} ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${GLUT_LIBRARIES} /usr/lib/libSOIL.so assimp ${GTK3_LIBRARIES} ${GLIBMM_LIBRARIES} ${GTKMM_LIBRARIES} ${CERES_LIBRARIES})



  

    




