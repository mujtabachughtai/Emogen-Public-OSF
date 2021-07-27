Blendshape model weights visualisation 2.0
Input: .csv with blendshape model weights (see example in /user_data/)
Output: rendering as .jpg image and (optionally) 3D mesh (.obj + .mtl files)

Input format:
sample.csv in /user_data/ folder is an example of the input file for the visualiser
The input file contains any number of blendshape vectors stacked up as rows (each 150 weights long). This way you can visualise a set of faces in batch.
You can also visualise a subset of the input file weights (from the beginning up to "total" specified in the configuration file)

There must be a comma separator after *each* blendshape weight, even the last one in the row
End-of-line character (CRLF) is optional, the visualiser is meant to ignore it if present (tested on Linux 18.04)

The easiest way to check your format is correct is by comparing weights printed in the command line to what is in your csv file. 

If all your faces come out as the neutral face (only 0 weight values are printed in the command line) or the renderings are obviously wrong, most likely there is an incompatibility in your csv file.

Contact the developer if you get stuck with the format.

Developer: 
Nadejda Roubtsova
Centre for the Analysis of Motion, Entertainment Research and Applications (CAMERA), University of Bath

Use the provided run_visualisation.sh script to run it
