sudo add-apt-repository universe

sudo apt-get clean
sudo apt-get update
sudo apt install  -y pkg-config
sudo apt-get install -y git
sudo apt-get install -y cmake

sudo apt-get install -y libjpeg-dev libpng-dev libtiff-dev

sudo add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main"
sudo apt update
sudo apt install -y libjasper1 libjasper-dev

sudo apt install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install -y libxvidcore-dev libx264-dev
sudo apt-get install -y python-dev python3-dev
sudo apt-get install -y libgtk-3-dev

# OpenCV
if [[ $(pkg-config --modversion opencv) != "3.2.0" ]]; then
echo "Either no OpenCV found or not the required version. Preparing to install OpenCV 3.2.0 ..."

sudo rm -r opencv
sudo rm -r opencv_contrib
git clone -b 3.2.0 https://github.com/opencv/opencv
git clone -b 3.2.0 https://github.com/opencv/opencv_contrib

root_folder=$PWD
cd opencv
sudo rm -r build
mkdir build
cd build

sudo cmake -DCMAKE_BUILD_TYPE=RELEASE \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DWITH_CUDA=OFF \
      -DOPENCV_EXTRA_MODULES_PATH=$root_folder/opencv_contrib/modules \
      -DOPENCV_ENABLE_NONFREE=ON \
      -DOPENCV_GENERATE_PKGCONFIG=ON ..

make -j7 

sudo make install

cd ../
cd ../
fi


# ASSIMP
if [[ $(pkg-config --modversion assimp) != "3.3.1" ]]; then
echo "Found ASSIMP version..."
echo $(pkg-config --modversion assimp)
echo "ASSIMP 3.3.1 not found. Preparing to install..."
sudo rm -r assimp-3.3.1
unzip assimp-3.3.1.zip
cd assimp-3.3.1
cmake -DASSIMP_BUILD_TESTS=OFF CMakeLists.txt -G 'Unix Makefiles' 
make 
sudo make install
cd ../
fi


# TODO: check whether ceres is installed
# CERES and dependencies
echo "Preparing to install Ceres 1.12.0.."
sudo apt-get -y install libgoogle-glog-dev
sudo apt -y install libeigen3-dev
sudo apt-get -y install libsuitesparse-dev
sudo apt-get -y install libatlas-base-dev

sudo rm -r ceres-bin
sudo rm -r ceres-solver-1.12.0
unzip ceres-solver-1.12.0.zip


mkdir ceres-bin
cd ceres-bin
echo $PWD
cmake -DBUILD_SHARED_LIBS=ON ../ceres-solver-1.12.0
make -j3
make test
sudo make install
cd ../

# OpenGL
# need to know what drivers are available on computer
sudo apt-get install mesa-utils
glxinfo | grep "OpenGL version"

# small openGL library for reading textures
sudo apt install -y libsoil-dev

sudo apt install -y libglfw3-dev
sudo apt install -y libglew-dev
sudo apt install -y freeglut3-dev

#GTKmm
sudo apt install -y libglibmm-2.4-dev
sudo apt install -y libgtkmm-3.0-dev
sudo apt install -y libglm-dev

#some additional fixes
sudo cp -r /usr/include/eigen3 /usr/local/include/
sudo cp -r /usr/local/include/eigen3/Eigen /usr/local/include/


#Now let's compile the tool itself
cd ../
cd EmoGen_tool
sudo rm -r build
mkdir build
cd build
cmake  ../
make

# Now let's build the program to convert a weights vector to an obj mesh
cd ../
cd ../
cd ApplyBlendweights/
sudo rm -r build
mkdir build
cd build
cmake ../
make 


