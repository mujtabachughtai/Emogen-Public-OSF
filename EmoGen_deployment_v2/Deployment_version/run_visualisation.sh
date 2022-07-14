#!/bin/bash

NUMBER_BLENDSHAPES=150

# SELECT MODEL: MALE OR FEMALE
model="MALE"

BLENDSHAPE_DIRECTORY=$PWD/data/$model"/"

if [ $model = "MALE" ]; then
   TEXTURE_FILE=$PWD/data/$model"/diffuse.bmp"
elif [ $model = "FEMALE" ]; then
   TEXTURE_FILE=$PWD/data/$model"/diffuse.png"
else  
   echo "Unknown model specified in the configuration file." 
   exit
fi

SHADER_DIRECTORY=$PWD/EmoGen_tool/common/

# important: format of csv file with blendshape weights (see readme.txt in ApplyBlendweights/ folder for details)
weights_FILE=$PWD/user_output/initialisation_transposed.csv

output_PATH=$PWD/user_output/

# number of faces from the csv file to be rendered: 1 to number of blendweight vectors in the csv file
total=10

# Need a 3D mesh (obj) as well as image? yes [true], no[false] Each mesh file is ~3.0 MB
obj_file=false

MESA_GL_VERSION_OVERRIDE=3.3 MESA_GLSL_VERSION_OVERRIDE=330 $PWD/ApplyBlendweights/build/go_3D $BLENDSHAPE_DIRECTORY $NUMBER_BLENDSHAPES $weights_FILE $output_PATH $TEXTURE_FILE $SHADER_DIRECTORY $total $obj_file


