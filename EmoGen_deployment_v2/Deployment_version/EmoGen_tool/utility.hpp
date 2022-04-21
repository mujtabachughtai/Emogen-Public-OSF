#ifndef UTILITY_HPP
#define UTILIT_HPP

#include <eigen3/Eigen/Eigen>
#include <eigen3/Eigen/Core>

// opencv
# include <opencv2/highgui/highgui.hpp>
# include "opencv2/imgproc/imgproc.hpp"
# include <opencv2/core/core.hpp>
# include "opencv2/calib3d/calib3d.hpp"
# include "opencv/cv.h"
# include "opencv2/features2d/features2d.hpp"
# include "opencv2/highgui/highgui.hpp"
# include "opencv2/opencv.hpp"
# include  <opencv2/tracking/tracker.hpp>
#include <opencv2/core/eigen.hpp>
#include <iostream>
#include <fstream>
#include <assimp/scene.h>
#include "data.hpp"
#include <map>


extern int GenNr_counter;
extern int NumberOfBlendshapes;
extern cv::Mat translation;    
extern cv::Mat rotation;       
extern double scale;    
extern bool doScale;
extern data * my_data;
extern cv::Mat_<double> allBlendShapeVertices3Nx1;
extern std::vector<unsigned int> indices_per_blendshape;
extern int numberOfvrtx;
extern std::string OUTPUT_DIRECTORY;
extern std::vector<int> sample_list;

class utility {

     public:

     utility(){};
   
     void compute_smooth_vertex_normals(aiMesh* mesh);

     cv::Mat read_csv(std::string input_csv);

     void write_session_to_csv_file(std::string filename);

     double procrustes(cv::Mat& X,cv::Mat& Y);
     double sumSquared( const cv::Mat& mat );

     int left_or_right(std::string blnd_str, std::size_t &found_special);
     bool seals_and_collisions(std::string blnd_str);
     bool find_head_motions(std::string blnd_str);

     cv::Mat read_barycentrics(std::string filename);
     cv::Mat get_anchor_point_cartersian_coordinates(aiMesh* mesh, cv::Mat &barycentrics, bool rotate);
     
  
     void apply_correctives(aiMesh* mesh, int choice_nr);
     void apply_head_motion(aiMesh* mesh, int choice_nr);
     void apply_any_set_of_blendshapes(aiMesh* mesh, int choice_nr, std::vector<int> blnd_nr_vec);
     void correct_lip_and_teeth_collisions(aiMesh* mesh, int choice_nr, bool &lip_collision, bool &teeth_collsion, bool as_checker);

     int test_all_triangles(cv::Mat &ptr, aiMesh* mesh, cv::Mat &direction, std::vector<unsigned int> &intersections, cv::Mat &true_intersections);
     void Barycentric(cv::Mat &p, cv::Mat &a, cv::Mat &b, cv::Mat &c, double &u, double &v, double &w);
   
     void generate_initialisation(cv::Mat& initialisation, std::vector<int> &vec_happy, std::vector<int> &vec_sad, 
                                     std::vector<int> &vec_angry, std::vector<int> &vec_fearful, std::map<int,int> &left_right_pairs /*, std::vector<int> &face_order*/);
     void enforce_symmetry(int order_ID, std::vector<int> &which_blendshapes, cv::Mat&initialisation, std::map<int,int> &left_right_pairs);
};
#endif
