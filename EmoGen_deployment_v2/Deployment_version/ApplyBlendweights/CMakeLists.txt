cmake_minimum_required(VERSION 2.8.4)

PROJECT(Visualise_solve)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp -std=c++11")
find_package(OpenCV REQUIRED)
find_package(assimp REQUIRED)
find_package(OpenGL)
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)
find_package(glfw3 REQUIRED)


include_directories(${OPENGL_INCLUDE_DIRS} ${GLEW_INCLUDE_DIRS})
add_executable(go_3D main.cpp ../EmoGen_tool/common/shader.cpp ../EmoGen_tool/common/texture.cpp) 
target_link_libraries(go_3D glfw ${OpenCV_LIBS} assimp ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${GLUT_LIBRARIES} /usr/lib/libSOIL.so )



  

    




