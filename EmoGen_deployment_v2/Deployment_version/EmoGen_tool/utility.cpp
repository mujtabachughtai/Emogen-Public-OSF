#include <eigen3/Eigen/Eigen>
#include "utility.hpp"
#include "ceres/ceres.h"
#include "CorrectiveWeightsAnalytic.cpp"
#include <random>
#include <chrono>

using ceres::Solver;
using namespace Eigen;

extern std::string OUTPUT_DIRECTORY;

void utility::compute_smooth_vertex_normals(aiMesh* mesh){

    // zero out all normals
    for (unsigned int vrtx_id = 0; vrtx_id < mesh->mNumVertices; vrtx_id++) mesh->mNormals[vrtx_id] = aiVector3D(0.0f,0.0f,0.0f); 


    for (unsigned int face_id = 0; face_id < mesh->mNumFaces; face_id++){
  

              const aiFace& face = mesh->mFaces[face_id]; 
	      aiVector3D a = mesh->mVertices[face.mIndices[0]]; 
              aiVector3D b = mesh->mVertices[face.mIndices[1]]; 
              aiVector3D c = mesh->mVertices[face.mIndices[2]]; 
 
              aiVector3D e1  = a - b;
              aiVector3D e2  = c - b;
              aiVector3D normal = -(e1 ^ e2); 
 
              mesh->mNormals[face.mIndices[0]] += normal;
              mesh->mNormals[face.mIndices[1]] += normal;
              mesh->mNormals[face.mIndices[2]] += normal;
			
     }

    for (unsigned int vrtx_id = 0; vrtx_id < mesh->mNumVertices; vrtx_id++) mesh->mNormals[vrtx_id].Normalize();

}



void utility::generate_initialisation(cv::Mat& initialisation, std::vector<int> &vec_happy, std::vector<int> &vec_sad, 
                                         std::vector<int> &vec_angry, std::vector<int> &vec_fearful, std::map<int,int> &left_right_pairs ) {


   cv::Mat init(10, NumberOfBlendshapes, CV_64F);
   init.setTo(0.0);
   init.copyTo(initialisation);
  
   unsigned seed_1 = std::chrono::system_clock::now().time_since_epoch().count();
   std::mt19937 generator_1(seed_1); 
   std::uniform_int_distribution<int> distribution_1(3, 8);

   int number_blnd_happy_1 = distribution_1(generator_1);
   int number_blnd_happy_2 = distribution_1(generator_1);
   int number_blnd_sad_1 = distribution_1(generator_1);
   int number_blnd_sad_2 = distribution_1(generator_1);
   int number_blnd_angry_1 = distribution_1(generator_1);
   int number_blnd_angry_2 = distribution_1(generator_1);
   int number_blnd_fearful_1 = distribution_1(generator_1);
   int number_blnd_fearful_2 = distribution_1(generator_1);
   int number_blnd_all = distribution_1(generator_1);
   
   unsigned seed_2 = seed_1;
   while(seed_2 == seed_1) seed_2 = std::chrono::system_clock::now().time_since_epoch().count();
   std::mt19937 generator_2(seed_2); 
   std::uniform_real_distribution<double> distribution_2(0.0, 1.0);

   std::vector<int> order_of_presentation;
   for (unsigned int i=0; i<10; ++i) order_of_presentation.push_back(i);

   unsigned seed_3 = seed_2;
   while(seed_3 == seed_2) seed_3 = std::chrono::system_clock::now().time_since_epoch().count();
   std::mt19937 generator_3(seed_3); 
   std::shuffle (order_of_presentation.begin(), order_of_presentation.end(), generator_3);
   // face_order = order_of_presentation;
  
   unsigned seed_happy = seed_3;
   while(seed_happy == seed_3) seed_happy = std::chrono::system_clock::now().time_since_epoch().count();
   std::mt19937 generator_happy(seed_happy); 
   std::uniform_int_distribution<int> distribution_happy(0, vec_happy.size() - 1);
   
   // happy 1
   std::vector<int> which_blnds;
   int order_ID = order_of_presentation[0];
   for(unsigned int blnd_nr = 0; blnd_nr < number_blnd_happy_1; ++blnd_nr) {
       double weight = distribution_2(generator_2);
       int blnd_subset_ID = distribution_happy(generator_happy);
       which_blnds.push_back(vec_happy[blnd_subset_ID]);
       initialisation.at<double>(order_ID, vec_happy[blnd_subset_ID]) = weight;
   }
  
   enforce_symmetry(order_ID, which_blnds, initialisation, left_right_pairs);

   // happy 2
   which_blnds.clear();
   order_ID = order_of_presentation[1];
   for(unsigned int blnd_nr = 0; blnd_nr < number_blnd_happy_2; ++blnd_nr) {
       double weight = distribution_2(generator_2);
       int blnd_subset_ID = distribution_happy(generator_happy);
       which_blnds.push_back(vec_happy[blnd_subset_ID]);
       initialisation.at<double>(order_ID, vec_happy[blnd_subset_ID]) = weight;

   }
   enforce_symmetry(order_ID, which_blnds, initialisation, left_right_pairs);

   unsigned seed_sad = std::chrono::system_clock::now().time_since_epoch().count();
   std::mt19937 generator_sad(seed_sad); 
   std::uniform_int_distribution<int> distribution_sad(0, vec_sad.size() - 1);

   // sad 1
   which_blnds.clear();
   order_ID = order_of_presentation[2];
   for(unsigned int blnd_nr = 0; blnd_nr < number_blnd_sad_1; ++blnd_nr) {
       double weight = distribution_2(generator_2);
       int blnd_subset_ID = distribution_sad(generator_sad);
       which_blnds.push_back(vec_sad[blnd_subset_ID]);
       initialisation.at<double>(order_ID, vec_sad[blnd_subset_ID]) = weight;
   }
   enforce_symmetry(order_ID, which_blnds, initialisation, left_right_pairs);

   // sad 2
   which_blnds.clear();
   order_ID = order_of_presentation[3];
   for(unsigned int blnd_nr = 0; blnd_nr < number_blnd_sad_2; ++blnd_nr) {
       double weight = distribution_2(generator_2);
       int blnd_subset_ID = distribution_sad(generator_sad);
       which_blnds.push_back(vec_sad[blnd_subset_ID]);
       initialisation.at<double>(order_ID, vec_sad[blnd_subset_ID]) = weight;

   }
   enforce_symmetry(order_ID, which_blnds, initialisation, left_right_pairs);

   unsigned seed_angry = std::chrono::system_clock::now().time_since_epoch().count();
   std::mt19937 generator_angry(seed_angry); 
   std::uniform_int_distribution<int> distribution_angry(0, vec_angry.size() - 1);

   // angry 1
   which_blnds.clear();
   order_ID = order_of_presentation[4];
   for(unsigned int blnd_nr = 0; blnd_nr < number_blnd_angry_1; ++blnd_nr) {
       double weight = distribution_2(generator_2);
       int blnd_subset_ID = distribution_angry(generator_angry);
       which_blnds.push_back(vec_angry[blnd_subset_ID]);
       initialisation.at<double>(order_ID, vec_angry[blnd_subset_ID]) = weight;
   }
   enforce_symmetry(order_ID, which_blnds, initialisation, left_right_pairs);

   // angry 2
   which_blnds.clear();
   order_ID = order_of_presentation[5];
   for(unsigned int blnd_nr = 0; blnd_nr < number_blnd_angry_2; ++blnd_nr) {
       double weight = distribution_2(generator_2);
       int blnd_subset_ID = distribution_angry(generator_angry);
       which_blnds.push_back(vec_angry[blnd_subset_ID]);
       initialisation.at<double>(order_ID, vec_angry[blnd_subset_ID]) = weight;
   }
   enforce_symmetry(order_ID, which_blnds, initialisation, left_right_pairs);

   unsigned seed_fearful = std::chrono::system_clock::now().time_since_epoch().count();
   std::mt19937 generator_fearful(seed_fearful); 
   std::uniform_int_distribution<int> distribution_fearful(0, vec_fearful.size() - 1);

   // fearful 1
   which_blnds.clear();
   order_ID = order_of_presentation[6];
   for(unsigned int blnd_nr = 0; blnd_nr < number_blnd_fearful_1; ++blnd_nr) {
       double weight = distribution_2(generator_2);
       int blnd_subset_ID = distribution_fearful(generator_fearful);
       which_blnds.push_back(vec_fearful[blnd_subset_ID]);
       initialisation.at<double>(order_ID, vec_fearful[blnd_subset_ID]) = weight;
   }
   enforce_symmetry(order_ID, which_blnds, initialisation, left_right_pairs);

   // fearful 2
   which_blnds.clear();
   order_ID = order_of_presentation[7];
   for(unsigned int blnd_nr = 0; blnd_nr < number_blnd_fearful_2; ++blnd_nr) {
       double weight = distribution_2(generator_2);
       int blnd_subset_ID = distribution_fearful(generator_fearful);
       which_blnds.push_back(vec_fearful[blnd_subset_ID]);
       initialisation.at<double>(order_ID, vec_fearful[blnd_subset_ID]) = weight;
   }
   enforce_symmetry(order_ID, which_blnds, initialisation, left_right_pairs);
   
   // freestyle
   unsigned seed_freestyle = std::chrono::system_clock::now().time_since_epoch().count();
   std::mt19937 generator_freestyle(seed_freestyle); 
   std::uniform_int_distribution<int> distribution_freestyle(0, sample_list.size() - 1);

   which_blnds.clear();
   order_ID = order_of_presentation[8];
   for(unsigned int blnd_nr = 0; blnd_nr < number_blnd_all; ++blnd_nr) {
       double weight = distribution_2(generator_2);
       int blnd_subset_ID = distribution_freestyle(generator_freestyle);
       which_blnds.push_back(sample_list[blnd_subset_ID]);
       initialisation.at<double>(order_ID, sample_list[blnd_subset_ID]) = weight;

   }
   enforce_symmetry(order_ID, which_blnds, initialisation, left_right_pairs);


}

void utility::enforce_symmetry(int order_ID, std::vector<int> &which_blendshapes, cv::Mat&initialisation, std::map<int,int> &left_right_pairs) {

	for (unsigned int wch_bld_nr = 0; wch_bld_nr < which_blendshapes.size(); ++wch_bld_nr) {

             for (std::map<int, int>::iterator it = left_right_pairs.begin(); it != left_right_pairs.end(); ++it) {

                  if ( it->first == which_blendshapes[wch_bld_nr] ) {

                      initialisation.at<double>(order_ID, it->second) =  initialisation.at<double>(order_ID, which_blendshapes[wch_bld_nr]);
                      break;

                  } else if (it->second == which_blendshapes[wch_bld_nr] ) {

                      initialisation.at<double>(order_ID, it->first) = initialisation.at<double>(order_ID, which_blendshapes[wch_bld_nr]);
                      break;

                 }

             }
       }
}



cv::Mat utility::read_csv(std::string input_csv){

	std::ifstream file(input_csv);
        if (!file) {std::cout << "Error: session initialisation file is not available" << std::endl; return cv::Mat();}
        int counter = 0;
        cv::Mat data_stream;
        std::string::size_type sz;   

        while ( file.good() ) {    
             
             std::string value;
	     getline ( file, value, ',' );


             std::size_t pos = value.find('\n');
   
             if ( pos != std::string::npos) {

                  std::string last_value_in_line = value.substr (0, pos);
                  data_stream.push_back(std::stod(last_value_in_line, &sz));
                  counter++;
               
                  std::string first_value_in_line = value.substr(pos + 1);
                  if (!first_value_in_line.empty()) {
                    data_stream.push_back(std::stod(first_value_in_line, &sz));
		    counter++;
                  }
                 

             } else  { 
               data_stream.push_back(std::stod(value, &sz)); 
               counter++; 
             }
             

         }

        return data_stream;
       
    };



double utility::procrustes(cv::Mat& X,cv::Mat& Y ){

    cv::Mat mu_x(1,3,CV_64F);
    mu_x.at<double>(0) = cv::mean(X.col(0))[0];
    mu_x.at<double>(1) = cv::mean(X.col(1))[0];
    mu_x.at<double>(2) = cv::mean(X.col(2))[0];

    cv::Mat X0 = X - repeat(mu_x, X.rows, 1);
 
    cv::Mat mu_y(1,3,CV_64F);
    mu_y.at<double>(0) = cv::mean(Y.col(0))[0];
    mu_y.at<double>(1) = cv::mean(Y.col(1))[0];
    mu_y.at<double>(2) = cv::mean(Y.col(2))[0];

    cv::Mat Y0      = Y - repeat(mu_y, Y.rows, 1);
 
    double ss_X      = sumSquared( X0 );
    double norm_X    = sqrt( ss_X );
    X0 = X0 / norm_X;
 
    double ss_Y      = sumSquared( Y0 );
    double norm_Y    = sqrt( ss_Y );
    Y0 = Y0 / norm_Y;

    cv::Mat A = X0.t() * Y0;

    cv::Mat U, s, Vt;
    cv::SVD::compute(A, s, U, Vt);

    cv::Mat V           = Vt.t();
    rotation  = V * U.t();

    double trace_TA = cv::sum(s)[0];

    if (doScale) {

      scale = trace_TA * norm_X/ norm_Y;
      
  
    } else {

      scale = 1.0;

    }
   
    translation = mu_x - scale *  mu_y * rotation; 
   
 
    return trace_TA; 
}; 


double utility::sumSquared( const cv::Mat& mat ) {

    cv::Mat temp;
    cv::pow( mat, 2.0, temp );
    cv::Scalar temp_scalar = cv::sum( temp );

    return temp_scalar[0];
};


int utility::left_or_right(std::string blnd_str, std::size_t &found_special ){

   std::size_t found_corrective = blnd_str.find("_");
   if (found_corrective != std::string::npos) return 0;

   std::size_t found_lft = blnd_str.find("lft");
   if (found_lft != std::string::npos) {found_special = found_lft; return 1;}

   std::size_t found_rgt = blnd_str.find("rgt");
   if (found_rgt != std::string::npos) {found_special = found_rgt; return 2;}

   return 0;

};

bool utility::seals_and_collisions(std::string blnd_str){

   std::size_t found_seal = blnd_str.find("Seal");
   std::size_t found_collision = blnd_str.find("Clsn");
   std::size_t found_neutral = blnd_str.find("neutral");
   std::size_t found_corrective = blnd_str.find("_");
  
   if (found_seal != std::string::npos || found_collision != std::string::npos || found_neutral != std::string::npos|| found_corrective != std::string::npos) return true;
   else return false;


};

bool utility::find_head_motions(std::string blnd_str){

   std::size_t found_head = blnd_str.find("head");

  
   if (found_head != std::string::npos) return true;
   else return false;


};

cv::Mat utility::read_barycentrics(std::string filename) {

    cv::Mat barycentrics;

    std::ifstream barycentrics_infi;
    barycentrics_infi.open(filename);
    if(!barycentrics_infi.is_open()) {
      return barycentrics;
    }

    while (!barycentrics_infi.eof() ){

        double face_id, coord_1, coord_2;
        barycentrics_infi >>  face_id;
        if ( barycentrics_infi.fail() ) return barycentrics;
        barycentrics_infi >>  coord_1;
        barycentrics_infi >>  coord_2;

        cv::Mat coord = (cv::Mat_<double>(1,3) << face_id,  coord_1,  coord_2);
        barycentrics.push_back(coord);
        

    }
    return barycentrics;

};

cv::Mat utility::get_anchor_point_cartersian_coordinates(aiMesh* mesh, cv::Mat &barycentrics, bool rotate){

        cv::Mat collision_anchor_neutral;
        for (unsigned int collision_nr = 0; collision_nr < barycentrics.rows; ++collision_nr){

		int face_id;
                double coord_1, coord_2;
		face_id = (int) barycentrics.at<double>(collision_nr, 0);
                coord_1 = barycentrics.at<double>(collision_nr, 1);
                coord_2 = barycentrics.at<double>(collision_nr, 2);	

	        const aiFace& face = mesh->mFaces[face_id]; 

	        cv::Mat face_vrtx_1, face_vrtx_2, face_vrtx_3;

	        aiVector3D pos = mesh->mVertices[face.mIndices[0] ]; 
	        face_vrtx_1 = (cv::Mat_<double>(1,3) << pos.x,  pos.y,  pos.z);

		pos = mesh->mVertices[face.mIndices[1] ]; 
		face_vrtx_2 = (cv::Mat_<double>(1,3) << pos.x,  pos.y,  pos.z);

		pos = mesh->mVertices[face.mIndices[2]]; 
		face_vrtx_3 = (cv::Mat_<double>(1, 3) << pos.x,  pos.y,  pos.z);

		cv::Mat ptt = coord_1 * face_vrtx_1 + coord_2 * face_vrtx_2 + (1 - coord_1 - coord_2) * face_vrtx_3;
            
                if (rotate)  ptt = scale * ptt * rotation +  translation;

                
                collision_anchor_neutral.push_back(ptt);
                
                               
            }

           return collision_anchor_neutral;

};


int utility::test_all_triangles(cv::Mat &ptr, aiMesh* mesh, cv::Mat &direction, std::vector<unsigned int> &intersections, cv::Mat &true_intersection){


   int intersections_pos = 0;
   int intersections_neg = 0;
   int numberOfintersections = 0;

   int nrOffaces = my_data -> lower_lip_coordinates.rows;

   double max_z_separation = 0.0;
   std::vector<cv::Mat> normals;
   std::vector<int> internal_intersections;
   for (unsigned int face_nr = 0; face_nr < nrOffaces; ++face_nr){

         int face_id = (int) my_data -> lower_lip_coordinates.at<double>(face_nr, 0);
         aiFace& face = mesh->mFaces[face_id]; 
         cv::Mat face_vrtx_1, face_vrtx_2, face_vrtx_3;
	 aiVector3D pos = mesh->mVertices[ face.mIndices[0] ]; 
	 face_vrtx_1 = (cv::Mat_<double>(1,3) << pos.x,  pos.y,  pos.z);
	 pos = mesh->mVertices[ face.mIndices[1] ]; 
	 face_vrtx_2 = (cv::Mat_<double>(1,3) << pos.x,  pos.y,  pos.z);
	 pos = mesh->mVertices[ face.mIndices[2] ]; 
	 face_vrtx_3 = (cv::Mat_<double>(1,3) << pos.x,  pos.y,  pos.z);
      
         cv::Mat face_normal = -(face_vrtx_1 - face_vrtx_2).cross(face_vrtx_3 - face_vrtx_2);
         face_normal = face_normal /cv::norm(face_normal);
         normals.push_back(face_normal);
         // parallel to face
         if (face_normal.dot(direction) == 0) continue;
   
         cv::Mat ptr_to_face = face_vrtx_1 - ptr;
         
         // coincidence to face
         if (face_normal.dot(-ptr_to_face) == 0) continue;

         double solution = face_normal.dot(ptr_to_face) / face_normal.dot(direction);

         cv::Mat intersection_ptr = ptr + solution * direction; 
         
           

         double u,v,w;
         Barycentric(intersection_ptr, face_vrtx_1,face_vrtx_2, face_vrtx_3, u,v,w);
         bool within_test = ((v+w) <= 1) && (v>=0 && v<=1) && (w>=0 && w<=1);       

         if ( within_test && solution > 0 ) { intersections_pos++; }
         if ( within_test && solution < 0 ) { 

            intersections_neg++; 
            if (fabs(solution) > max_z_separation) {

               max_z_separation = fabs(solution);
               intersection_ptr.copyTo(true_intersection);
               if( intersections.size() == 0) {
                    intersections.push_back(face_nr); 
               } else {
                 
                 intersections[0] = face_nr;
               }

           }

           internal_intersections.push_back(face_nr);

         }

   }


   //numberOfintersections = std::max(intersections_pos, intersections_neg);
   
   if (intersections_pos == 1 && intersections_neg == 1) numberOfintersections =1;
   if (intersections_neg == 2) {
         // major change
         double check = normals[internal_intersections[0]].dot(normals[internal_intersections[1]]);
         if(check < 0.0) numberOfintersections = 2;

   }
  
   return numberOfintersections;

}


void utility::Barycentric(cv::Mat &p, cv::Mat &a, cv::Mat &b, cv::Mat &c, double &u, double &v, double &w)
{
    cv::Mat v0 = b - a;
    cv::Mat v1 = c - a;
    cv::Mat v2 = p - a;
    double d00 = v0.dot(v0);
    double d01 = v0.dot(v1);
    double d11 = v1.dot(v1);
    double d20 = v2.dot(v0);
    double d21 = v2.dot(v1);
    double denom = d00 * d11 - d01 * d01;
    v = (d11 * d20 - d01 * d21) / denom;
    w = (d00 * d21 - d01 * d20) / denom;
    u = 1.0 - v - w;
}

void utility::apply_head_motion(aiMesh* mesh, int choice_nr) {


   for (unsigned int head_motion_blnd_nr = 0; head_motion_blnd_nr < my_data -> head_motion_blnds.size(); ++head_motion_blnd_nr){

       double head_motion_weight = my_data->weights_current_generation[choice_nr][my_data -> head_motion_blnds[head_motion_blnd_nr]]; 

        for (unsigned int k=0; k< mesh->mNumVertices; k++){

		 mesh->mVertices[k].x = mesh->mVertices[k].x + head_motion_weight
		                                             * allBlendShapeVertices3Nx1(my_data -> head_motion_blnds[head_motion_blnd_nr] * 3 * numberOfvrtx + 3 * k);
		 mesh->mVertices[k].y = mesh->mVertices[k].y + head_motion_weight 
		                                             * allBlendShapeVertices3Nx1(my_data -> head_motion_blnds[head_motion_blnd_nr] * 3 * numberOfvrtx + 3 * k + 1);
		 mesh->mVertices[k].z = mesh->mVertices[k].z + head_motion_weight 
                                                             * allBlendShapeVertices3Nx1(my_data -> head_motion_blnds[head_motion_blnd_nr] * 3 * numberOfvrtx + 3 * k + 2);
         }
   }

}

void utility::apply_any_set_of_blendshapes(aiMesh* mesh, int choice_nr, std::vector<int> blnd_nr_vec) {

     for (unsigned int blnd_nr = 0; blnd_nr < blnd_nr_vec.size(); ++blnd_nr) {
          
          double weight = my_data->weights_current_generation[choice_nr][blnd_nr_vec[blnd_nr]]; 

           for (unsigned int k=0; k< mesh->mNumVertices; k++){

		 mesh->mVertices[k].x = mesh->mVertices[k].x + weight
		                                             * allBlendShapeVertices3Nx1(blnd_nr_vec[blnd_nr] * 3 * numberOfvrtx + 3 * k);
		 mesh->mVertices[k].y = mesh->mVertices[k].y + weight 
		                                             * allBlendShapeVertices3Nx1(blnd_nr_vec[blnd_nr] * 3 * numberOfvrtx + 3 * k + 1);
		 mesh->mVertices[k].z = mesh->mVertices[k].z + weight 
                                                             * allBlendShapeVertices3Nx1(blnd_nr_vec[blnd_nr] * 3 * numberOfvrtx + 3 * k + 2);
           }

     }

}


void utility::correct_lip_and_teeth_collisions(aiMesh* mesh, int choice_nr, bool &lip_collision, bool &teeth_collsion, bool as_checker) {


           /*std::string cloud_top_lip_file = OUTPUT_DIRECTORY +"top_lip"+std::to_string(choice_nr + 1)+".ply";
           
           std::ofstream cloud_top_lip;
           cloud_top_lip.open(cloud_top_lip_file);
           cloud_top_lip << "ply" << std::endl;
           cloud_top_lip << "format ascii 1.0" << std::endl;

           std::string cloud_bottom_lip_file = OUTPUT_DIRECTORY +"bottom_lip"+std::to_string(choice_nr + 1)+".ply";
          
           std::ofstream cloud_bottom_lip;
           cloud_bottom_lip.open(cloud_bottom_lip_file);
           cloud_bottom_lip << "ply" << std::endl;
           cloud_bottom_lip << "format ascii 1.0" << std::endl;


           std::string cloud_teeth_file = OUTPUT_DIRECTORY +"teeth"+std::to_string(choice_nr +1)+".ply";
           std::ofstream cloud_teeth;
           cloud_teeth.open(cloud_teeth_file);
           cloud_teeth << "ply" << std::endl;
           cloud_teeth << "format ascii 1.0" << std::endl;

           std::string cloud_bottom_lip_teeth_file = OUTPUT_DIRECTORY +"bottom_lip_teeth"+std::to_string(choice_nr + 1)+".ply";
           std::ofstream cloud_bottom_lip2;
           cloud_bottom_lip2.open(cloud_bottom_lip_teeth_file);
           cloud_bottom_lip2 << "ply" << std::endl;
           cloud_bottom_lip2 << "format ascii 1.0" << std::endl; */


           std::vector<double> y_offsets, z_offsets;
           cv::Mat all_lower_lip_points;
           int numberOflowerlip_ptrs = my_data -> lower_lip_coordinates.rows;
            
           for (unsigned int lwr_lip_ptr_nr = 0; lwr_lip_ptr_nr < numberOflowerlip_ptrs; ++lwr_lip_ptr_nr) {

                     int face_id = (int) my_data -> lower_lip_coordinates.at<double>(lwr_lip_ptr_nr, 0);
		     double coord_1 =    my_data -> lower_lip_coordinates.at<double>(lwr_lip_ptr_nr, 1);
                     double coord_2 =    my_data -> lower_lip_coordinates.at<double>(lwr_lip_ptr_nr, 2);

		     aiFace& face = mesh->mFaces[face_id]; 
		     cv::Mat face_vrtx_1, face_vrtx_2, face_vrtx_3;

		     aiVector3D pos = mesh->mVertices[ face.mIndices[0] ]; 
		     face_vrtx_1 = (cv::Mat_<double>(1,3) << pos.x,  pos.y,  pos.z);
	   
		     pos = mesh->mVertices[ face.mIndices[1] ]; 
		     face_vrtx_2 = (cv::Mat_<double>(1,3) << pos.x,  pos.y,  pos.z);

		     pos = mesh->mVertices[ face.mIndices[2] ]; 
		     face_vrtx_3 = (cv::Mat_<double>(1,3) << pos.x,  pos.y,  pos.z);

	     
		     cv::Mat ptt = (coord_1 * face_vrtx_1 + coord_2 * face_vrtx_2 + (1 - coord_1 - coord_2) * face_vrtx_3);


                     all_lower_lip_points.push_back(ptt);
               }


             int numOfcollision_ptrs =  my_data -> collision_anchor_coordinates.rows;
  
             cv::Mat all_anchors(numOfcollision_ptrs, 3, CV_64F);
             all_anchors.setTo(0.0);

             
             std::vector<cv::Point2i> anchors_to_cut_point_face_nr(numOfcollision_ptrs);
             std::vector<cv::Mat> true_intersections_per_lip_point(numOfcollision_ptrs, cv::Mat());

             int lips_count = 0;
             int active_point_count = 0;

             bool collision_lips = false;

             bool zone1 = false;
             bool zone2 = false;
             bool zone3 = false;
             bool zone4 = false;
             
             #pragma omp parallel for 
             for (unsigned int anchor_nr = 0; anchor_nr < numOfcollision_ptrs; ++anchor_nr) {

                     int face_id = (int) my_data -> collision_anchor_coordinates.at<double>(anchor_nr, 0);
		     double coord_1 =    my_data -> collision_anchor_coordinates.at<double>(anchor_nr, 1);
                     double coord_2 =    my_data -> collision_anchor_coordinates.at<double>(anchor_nr, 2);

		     aiFace& face = mesh->mFaces[face_id]; 
		     cv::Mat face_vrtx_1, face_vrtx_2, face_vrtx_3;

		     aiVector3D pos = mesh->mVertices[ face.mIndices[0] ]; 
		     face_vrtx_1 = (cv::Mat_<double>(1,3) << pos.x,  pos.y,  pos.z);
	   
		     pos = mesh->mVertices[ face.mIndices[1] ]; 
		     face_vrtx_2 = (cv::Mat_<double>(1,3) << pos.x,  pos.y,  pos.z);

		     pos = mesh->mVertices[ face.mIndices[2] ]; 
		     face_vrtx_3 = (cv::Mat_<double>(1,3) << pos.x,  pos.y,  pos.z);

	     
		     cv::Mat ptt = (coord_1 * face_vrtx_1 + coord_2 * face_vrtx_2 + (1 - coord_1 - coord_2) * face_vrtx_3);

                     ptt.copyTo(all_anchors.row(anchor_nr));
                  
                     cv::Point2i pair;
                     pair.x = anchor_nr;
                     pair.y = -1;
                     

                     std::vector<unsigned int> face_intersections_z, face_intersections_y;
                     cv::Mat true_intersection_z, true_intersection_y;
                     cv::Mat direction = (cv::Mat_<double>(1,3) << 0.0,  -1.0,  0.0);
                     int numberOfintersections_z = test_all_triangles(ptt, mesh, direction, face_intersections_z, true_intersection_z);

                     if (face_intersections_z.size() > 0) {
                    
                        pair.y = face_intersections_z[0];
                        true_intersections_per_lip_point[anchor_nr] = true_intersection_z;
                        #pragma omp atomic
                        active_point_count++;

                      } 

		      direction.setTo(0.0);
		      direction = (cv::Mat_<double>(1,3) << 0.0,  0.0,  1.0);
		      int numberOfintersections_y = test_all_triangles(ptt, mesh, direction, face_intersections_y, true_intersection_y);
		       

                      if(numberOfintersections_y == 1 || numberOfintersections_z == 1) { 
                           #pragma omp atomic
                           lips_count++; 
                          
                           if (std::find(my_data->zone1.begin(), my_data->zone1.end(), anchor_nr) != my_data->zone1.end()){

                              #pragma omp atomic write
                              zone1 = true;

                           } else if (std::find(my_data->zone2.begin(), my_data->zone2.end(), anchor_nr) != my_data->zone2.end()) {

                              #pragma omp atomic write
                              zone2 = true;

                           } else if (std::find(my_data->zone3.begin(), my_data->zone3.end(), anchor_nr) != my_data->zone3.end()) {

                              #pragma omp atomic write
                              zone3 = true;

                           } else if (std::find(my_data->zone4.begin(), my_data->zone4.end(), anchor_nr) != my_data->zone4.end()) {

                              #pragma omp atomic write
                              zone4 = true;
                           } else {
                              std::cout << "Error: Not any lip zone" << std::endl;
                           }
                           

                      }
		      anchors_to_cut_point_face_nr[anchor_nr] = pair;
                      

              }


             if (lips_count > 0) collision_lips = true;
             

            /* cloud_top_lip << "element vertex " << active_point_count << std::endl;
             cloud_top_lip << "property float x" << std::endl;
	     cloud_top_lip << "property float y" << std::endl;
	     cloud_top_lip << "property float z" << std::endl;
	     cloud_top_lip << "end_header" << std::endl;

             cloud_bottom_lip << "element vertex " << active_point_count << std::endl;
             cloud_bottom_lip << "property float x" << std::endl;
	     cloud_bottom_lip << "property float y" << std::endl;
	     cloud_bottom_lip << "property float z" << std::endl;
	     cloud_bottom_lip << "end_header" << std::endl; */
            
              
             int numOfteeth_ptrs =  my_data -> collision_anchor_coordinates_teeth.rows;
             cv::Mat all_teeth_anchors(numOfteeth_ptrs, 3, CV_64F);
             all_teeth_anchors.setTo(0.0);
             
             std::vector<cv::Point2i> anchors_to_cut_point_face_nr2(numOfteeth_ptrs);
             std::vector<cv::Mat>true_intersections_per_teeth_point(numOfteeth_ptrs, cv::Mat());

             int teeth_count = 0;
             int count_outside = 0;
             int count_inside = 0;
             active_point_count = 0;
            
             
             bool collision_teeth = false;
             #pragma omp parallel for 
             for(unsigned int teeth_ptr_nr = 0; teeth_ptr_nr < numOfteeth_ptrs; ++teeth_ptr_nr) {
                     
                     int face_id = (int) my_data -> collision_anchor_coordinates_teeth.at<double>(teeth_ptr_nr, 0);
		     double coord_1 = my_data -> collision_anchor_coordinates_teeth.at<double>(teeth_ptr_nr, 1);
                     double coord_2 = my_data -> collision_anchor_coordinates_teeth.at<double>(teeth_ptr_nr, 2);

		     aiFace& face = mesh->mFaces[face_id]; 
		     cv::Mat face_vrtx_1, face_vrtx_2, face_vrtx_3;

		     aiVector3D pos = mesh->mVertices[ face.mIndices[0] ]; 
		     face_vrtx_1 = (cv::Mat_<double>(1,3) << pos.x,  pos.y,  pos.z);
	   
		     pos = mesh->mVertices[ face.mIndices[1] ]; 
		     face_vrtx_2 = (cv::Mat_<double>(1,3) << pos.x,  pos.y,  pos.z);

		     pos = mesh->mVertices[ face.mIndices[2] ]; 
		     face_vrtx_3 = (cv::Mat_<double>(1,3) << pos.x,  pos.y,  pos.z);

	     
		     cv::Mat ptt_teeth = (coord_1 * face_vrtx_1 + coord_2 * face_vrtx_2 + (1 - coord_1 - coord_2) * face_vrtx_3);
                    

                     ptt_teeth.copyTo(all_teeth_anchors.row(teeth_ptr_nr));
                      
                     cv::Point2i pair;
                     pair.x = teeth_ptr_nr;
                     pair.y = -1;
                     cv::Mat true_intersection_z, true_intersection_y;
                     std::vector<unsigned int> face_intersections_z, face_intersections_y;
	             cv::Mat direction = (cv::Mat_<double>(1,3) << 0.0,  -1.0,  0.0);
	             int numberOfintersections_z = test_all_triangles(ptt_teeth, mesh, direction, face_intersections_z, true_intersection_z);
                       

	             direction.setTo(0.0);
	             direction = (cv::Mat_<double>(1,3) << 0.0,  0.0,  1.0);
            
                   //  direction = -(face_vrtx_1 - face_vrtx_2).cross(face_vrtx_3 - face_vrtx_2);
                   //  direction = direction / cv::norm(direction);
                     
               
	            int numberOfintersections_y = test_all_triangles(ptt_teeth, mesh, direction, face_intersections_y, true_intersection_y);

                    if (face_intersections_y.size() > 0) {  // addition numberOfintersections_y > 0 ?

                          pair.y = face_intersections_y[0];
                          true_intersections_per_teeth_point[teeth_ptr_nr] = true_intersection_y;
                          #pragma omp atomic
                          active_point_count++;

                    } 

                    // second major change >>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	            if(numberOfintersections_y == 1 || numberOfintersections_z == 1 ) { 

                        #pragma omp atomic
                        teeth_count++; 
                        #pragma omp atomic
                        count_inside++; 

                    } 

                    if(numberOfintersections_y == 2) {

                       #pragma omp atomic
                       count_outside++;

                    }
        
                    anchors_to_cut_point_face_nr2[teeth_ptr_nr] = pair;
                 
                }
                 
                if (count_inside > 0 && count_outside > 0 ) collision_teeth = true;

                 
                 lip_collision = collision_lips;
                 teeth_collsion = collision_teeth;
               
                 if (as_checker) {   
                     
                     if (!( (zone2 && zone3) ) ) lip_collision = false; 
                     return;
                 }
                
                

               /* cloud_teeth << "element vertex " << active_point_count << std::endl;
		cloud_teeth << "property float x" << std::endl;
		cloud_teeth << "property float y" << std::endl;
		cloud_teeth << "property float z" << std::endl;
		cloud_teeth << "end_header" << std::endl;

		cloud_bottom_lip2 << "element vertex " << active_point_count << std::endl;
		cloud_bottom_lip2 << "property float x" << std::endl;
		cloud_bottom_lip2 << "property float y" << std::endl;
		cloud_bottom_lip2 << "property float z" << std::endl;
	        cloud_bottom_lip2 << "end_header" << std::endl; */
            

                //if (!collision_lips &&  !collision_teeth) {std::cout << "No collisions detected" << std::endl; return;}
                
                cv::Mat A_local;
               
	        for (unsigned int anchor_nr_t = 0; anchor_nr_t < numOfcollision_ptrs; ++anchor_nr_t) {
	        
	                cv::Mat top_ptr = all_anchors.row(anchor_nr_t);

			if ( anchors_to_cut_point_face_nr[anchor_nr_t].y == -1) { 

				  z_offsets.push_back(0.0); 
				
	                          cv::Mat row(1, 8, CV_64F);
				  row.setTo(0.0);
                                
				  A_local.push_back(row);
				  continue; 

			 } 

			 cv::Mat bottom_ptr = true_intersections_per_lip_point[anchor_nr_t];
                        
			 cv::Mat difference_vector =  -(top_ptr - bottom_ptr);  

	                 if (difference_vector.at<double>(1) < 0 ) {

			         //cloud_top_lip << top_ptr.at<double>(0) << " " << top_ptr.at<double>(1) << " " << top_ptr.at<double>(2) << std::endl;
                                 //cloud_bottom_lip << bottom_ptr.at<double>(0) << " " << bottom_ptr.at<double>(1) << " " << bottom_ptr.at<double>(2) << std::endl;
			          
                                 z_offsets.push_back(0.0); 
			         cv::Mat row(1, 8, CV_64F);
				 row.setTo(0.0);
			         A_local.push_back(row);
			         continue; 

	                  }

                         // change
			 if (!collision_lips) z_offsets.push_back(0.0);
			 else z_offsets.push_back(difference_vector.at<double>(1));

                        
                            
                        //cloud_top_lip << top_ptr.at<double>(0) << " " << top_ptr.at<double>(1) << " " << top_ptr.at<double>(2) << std::endl;
                        //cloud_bottom_lip << bottom_ptr.at<double>(0) << " " << bottom_ptr.at<double>(1) << " " << bottom_ptr.at<double>(2) << std::endl;  

			cv::Mat row(1, 8, CV_64F);
			row.setTo(0.0);

			for (unsigned int blnd_collision_nr = 0; blnd_collision_nr < 4; ++blnd_collision_nr) {

                             if(collision_lips) {

				  row.at<double>(blnd_collision_nr) = (my_data -> collision_deviations.at<double>(blnd_collision_nr * numOfcollision_ptrs  
                                                                      + anchor_nr_t, 1)
					                              - my_data -> collision_deviations_lwr_lip.at<double>(blnd_collision_nr * numberOflowerlip_ptrs 
	                                                              + anchors_to_cut_point_face_nr[anchor_nr_t].y, 1));  
                                   
                                  

	                          if(fabs(row.at<double>(blnd_collision_nr)) < 1e-1) row.at<double>(blnd_collision_nr) = 0.0;

                                 
                             
                             } else row.at<double>(blnd_collision_nr) = 0.0;

			  } 

	                 for (unsigned int blnd_collision_nr = 0; blnd_collision_nr < 4; ++blnd_collision_nr) {
				      
	                     if(collision_teeth) {

			           row.at<double>(blnd_collision_nr + 4) = (my_data -> collision_deviations2.at<double>(blnd_collision_nr * numOfcollision_ptrs  
                                                                            + anchor_nr_t, 1)
							              - my_data -> collision_deviations_lwr_lip_teeth.at<double>(blnd_collision_nr * numberOflowerlip_ptrs 
	                                                              + anchors_to_cut_point_face_nr[anchor_nr_t].y, 1)); 

                                   if (fabs(row.at<double>(blnd_collision_nr + 4)) < 1e-1)  row.at<double>(blnd_collision_nr + 4) = 0.0;  
                                   

	                     } else row.at<double>(blnd_collision_nr + 4) = 0.0;

			 }
 
			 A_local.push_back(row); 


	       } 
                   
               int constraints_lips = A_local.rows;

	       for (unsigned int anchor_nr_teeth = 0; anchor_nr_teeth < numOfteeth_ptrs; ++anchor_nr_teeth) {

			       cv::Mat teeth_ptr = all_teeth_anchors.row(anchor_nr_teeth);

			       if ( anchors_to_cut_point_face_nr2[anchor_nr_teeth].y == -1) { 

				          y_offsets.push_back(0.0);
				          cv::Mat row(1, 8, CV_64F);
					  row.setTo(0.0);
		                          A_local.push_back(row);
                                           
				          continue; 

				 } 
                                 
				 
                                    cv::Mat bottom_ptr = true_intersections_per_teeth_point[anchor_nr_teeth];

				 cv::Mat difference_vector =  (teeth_ptr -bottom_ptr); 

		                 if (difference_vector.at<double>(2) < 0 ) {

		                        // cloud_teeth << teeth_ptr.at<double>(0) << " " << teeth_ptr.at<double>(1) << " " << teeth_ptr.at<double>(2) << std::endl;
		                        // cloud_bottom_lip2 << bottom_ptr.at<double>(0) << " " << bottom_ptr.at<double>(1) << " " << bottom_ptr.at<double>(2) << std::endl;
				          
				          y_offsets.push_back(0.0); 
				          cv::Mat row(1, 8, CV_64F);
					  row.setTo(0.0);
				          A_local.push_back(row);
				          continue; 

	 
		                 }

                                 // change
                                 if (!collision_teeth) y_offsets.push_back(0.0);
				 else y_offsets.push_back(difference_vector.at<double>(2));
                                  
                                 //cloud_teeth << teeth_ptr.at<double>(0) << " " << teeth_ptr.at<double>(1) << " " << teeth_ptr.at<double>(2) << std::endl;
                                 //cloud_bottom_lip2 << bottom_ptr.at<double>(0) << " " << bottom_ptr.at<double>(1) << " " << bottom_ptr.at<double>(2) << std::endl;  

				 cv::Mat row(1,8, CV_64F);
				 row.setTo(0.0);

				 for (unsigned int blnd_collision_nr = 0; blnd_collision_nr < 4; ++blnd_collision_nr) {
					      
		                          if(collision_lips) { 

					       row.at<double>(blnd_collision_nr) = my_data -> collision_deviations_lwr_lip.at<double>
		                                                             (blnd_collision_nr * numberOflowerlip_ptrs  + anchors_to_cut_point_face_nr2[anchor_nr_teeth].y, 2)   
		                                                             - my_data -> collision_deviations_teeth2.at<double>(blnd_collision_nr * numOfteeth_ptrs + 
				                                              anchor_nr_teeth, 2);

                                               if (fabs(row.at<double>(blnd_collision_nr)) < 1e-1) row.at<double>(blnd_collision_nr) = 0.0;

		                           } else {

		                               row.at<double>(blnd_collision_nr) = 0.0;

		                           }
		                        
				  }

				 for (unsigned int blnd_collision_nr = 0; blnd_collision_nr < 4; ++blnd_collision_nr) {

                                      if (collision_teeth) {
					       
				               row.at<double>(blnd_collision_nr + 4) =  my_data -> collision_deviations_lwr_lip_teeth.at<double>
		                                                             (blnd_collision_nr * numberOflowerlip_ptrs + anchors_to_cut_point_face_nr2[anchor_nr_teeth].y,2);   
		                                                              - my_data -> collision_deviations_teeth.at<double>(blnd_collision_nr * numOfteeth_ptrs + 
				                                              anchor_nr_teeth, 2);

                                               if (fabs(row.at<double>(blnd_collision_nr + 4)) < 1e-1) row.at<double>(blnd_collision_nr + 4) = 0.0;


                                       } else  row.at<double>(blnd_collision_nr + 4) = 0.0;
         
				  } 
                                  
                                  A_local.push_back(row);
                             
		      }  
               
             cv::Mat b;
              
             for (unsigned int ptr_nr = 0; ptr_nr < z_offsets.size(); ++ptr_nr)  b.push_back(z_offsets[ptr_nr]); 
             for (unsigned int ptr_nr = 0; ptr_nr < y_offsets.size(); ++ptr_nr)  b.push_back(y_offsets[ptr_nr]); 
 

              if (cv::countNonZero(b) >= 8) {


		      cv::Mat_<double> collision_weights = cv::Mat(8,1, CV_64F, double(0.0));
		      double * collision_weights_ptr = (double *) collision_weights.data;

		      std::vector<CorrectiveWeightsAnalytic*> problems_collisions;
		      std::vector<ceres::ResidualBlockId> residual_block_ids;
		      ceres::Problem problem_collisions;
		      
		      
		       for (unsigned int constr_nr = 0; constr_nr < A_local.rows; ++constr_nr) {

                                 bool lip_constraint = false;
                                 if (constr_nr < constraints_lips) lip_constraint = true; 
		                 if ( cv::countNonZero(A_local.row(constr_nr)) == 0 ) continue;
                              
		                 cv::Mat row;
		                 A_local.row(constr_nr).copyTo(row);
		                 ceres::CostFunction* cost_function = new CorrectiveWeightsAnalytic(row, b.at<double>(constr_nr), lip_constraint);
				 ceres::ResidualBlockId block_id = problem_collisions.AddResidualBlock(cost_function, NULL, collision_weights_ptr);  
				 residual_block_ids.push_back(block_id);

		       }

		       for( int i = 0; i < 8; ++i) {

				problem_collisions.SetParameterLowerBound(collision_weights_ptr, i, 0);
		                problem_collisions.SetParameterUpperBound(collision_weights_ptr, i, 1);
			 
		       } 

		       Solver::Options options; 
		       options.max_num_iterations = 1000;
		       options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
		       options.use_nonmonotonic_steps= false;
		       options.preconditioner_type = ceres::PreconditionerType::SCHUR_JACOBI;
		       options.use_explicit_schur_complement = true;   
		       options.minimizer_type = ceres::MinimizerType::TRUST_REGION;
		       options.num_linear_solver_threads = 1;
		       options.minimizer_progress_to_stdout=false;		  
		       options.num_threads = 100;  
                            
		       Solver::Summary summary;

		       ceres::Solve(options, &problem_collisions, &summary);      
                       //std::cout  << summary.FullReport() << std::endl;

                       cv::Mat solution;
                       collision_weights.copyTo(solution);

		       for (unsigned int collision_blnd_nr = 0; collision_blnd_nr < 8; ++collision_blnd_nr) {

		              double corrective_weight = solution.at<double>(collision_blnd_nr);

		              if (corrective_weight == 0.0) continue;

                              std::map<int, std::string>::iterator it_blnd;
                              if (collision_blnd_nr < 4) it_blnd = my_data -> anchor_pair_nr_to_collision_blnd.find(collision_blnd_nr);
			      else  { it_blnd = my_data -> teeth_anchor_pair_nr_to_collision_blnd.find(collision_blnd_nr - 4);}

			      std::string corrective = it_blnd -> second;
			      std::map<std::string, int>::iterator it_blnd_nr = my_data->correctives.find(corrective);
			      int blnd_nr = it_blnd_nr -> second;
                              

                              my_data->weights_current_generation[choice_nr][blnd_nr] = corrective_weight;

                              for (unsigned int k=0; k< mesh->mNumVertices; k++){

					 mesh->mVertices[k].x = mesh->mVertices[k].x + corrective_weight
					                                             * allBlendShapeVertices3Nx1(blnd_nr * 3 * numberOfvrtx + 3 * k);
					 mesh->mVertices[k].y = mesh->mVertices[k].y + corrective_weight 
					                                             * allBlendShapeVertices3Nx1(blnd_nr * 3 * numberOfvrtx + 3 * k + 1);
					 mesh->mVertices[k].z = mesh->mVertices[k].z + corrective_weight 
		                                                                     * allBlendShapeVertices3Nx1(blnd_nr * 3 * numberOfvrtx + 3 * k + 2);

			      }


		       }

                    
                      
               
             } 

       

        /* cloud_top_lip.close();
         cloud_bottom_lip.close();
         cloud_teeth.close();
         cloud_bottom_lip2.close();*/

}




void utility::apply_correctives(aiMesh* mesh, int choice_nr){

  for (std::map<int, std::vector<double>>::iterator it_blnd_nr  =  my_data->correctives_to_activation_vector.begin();
                                                    it_blnd_nr != my_data->correctives_to_activation_vector.end(); ++it_blnd_nr) {
            
                 
          int blnd_nr = it_blnd_nr -> first;

          if  (my_data->weights_current_generation[choice_nr][blnd_nr] != 0.0) continue;

          std::vector<double> activation_vector = it_blnd_nr -> second;
          std::vector<double> blendweight_vector = my_data->weights_current_generation[choice_nr];

          cv::Mat activation_vector_mat(activation_vector.size(),1, CV_64F);
          activation_vector_mat.setTo(0.0);

          cv::Mat blendweight_vector_mat(blendweight_vector.size(), 1, CV_64F);
          blendweight_vector_mat.setTo(0.0);

          memcpy(activation_vector_mat.data, activation_vector.data(), activation_vector.size()*sizeof(double));
          memcpy(blendweight_vector_mat.data, blendweight_vector.data(), blendweight_vector.size()*sizeof(double));

          cv::Mat activated_blendweight_vector_mat = activation_vector_mat.mul(blendweight_vector_mat);
         
          if (cv::countNonZero(activated_blendweight_vector_mat) != cv::countNonZero(activation_vector_mat)) continue;

          
          activated_blendweight_vector_mat.setTo(1,  activated_blendweight_vector_mat ==0);
          double corrective_weight = 1;
          for (unsigned int weight_nr=0; weight_nr < blendweight_vector.size(); ++weight_nr) corrective_weight *= activated_blendweight_vector_mat.at<double>(weight_nr);


          for (unsigned int k=0; k< mesh->mNumVertices; k++){

                 
		 mesh->mVertices[k].x = mesh->mVertices[k].x + corrective_weight
		                                             * allBlendShapeVertices3Nx1(blnd_nr * 3 * numberOfvrtx + 3 * k);
		 mesh->mVertices[k].y = mesh->mVertices[k].y + corrective_weight 
		                                             * allBlendShapeVertices3Nx1(blnd_nr * 3 * numberOfvrtx + 3 * k + 1);
		 mesh->mVertices[k].z = mesh->mVertices[k].z + corrective_weight 
                                                             * allBlendShapeVertices3Nx1(blnd_nr * 3 * numberOfvrtx + 3 * k + 2);

	   }


          my_data->weights_current_generation[choice_nr][blnd_nr] = corrective_weight;
          

  }
		                

  return;
}


void utility::write_session_to_csv_file(std::string filename){

      if (my_data -> full_account.rows == 0 ) return;
   
      std::ofstream session_file;
      if (GenNr_counter == 0)  session_file.open(filename, std::ofstream::out | std::ofstream::trunc);
      else session_file.open(filename, std::ofstream::out | std::ofstream::app);
  
              
      session_file << "Generation number,"<< GenNr_counter << std::endl; 
      session_file <<"elite face,,"; 
      for (int col_nr = 0; col_nr < 9; ++col_nr) session_file << my_data -> full_account.at<double>(GenNr_counter * (2 + NumberOfBlendshapes),col_nr) << ",";
      session_file <<my_data -> full_account.at<double>(GenNr_counter * (2 + NumberOfBlendshapes), 9)<< std::endl;
      session_file <<"selected faces,,";
      for (int col_nr = 0; col_nr < 9; ++col_nr) session_file << my_data -> full_account.at<double>(GenNr_counter * (2 + NumberOfBlendshapes) + 1,col_nr) << ",";
      session_file <<my_data -> full_account.at<double>(GenNr_counter * (2 + NumberOfBlendshapes) + 1, 9)<< std::endl;
      

      for (int blnd_nr = 0; blnd_nr < NumberOfBlendshapes; ++blnd_nr) {

                  std::map<std::string, int>::iterator it;
	          for (it = my_data -> blendshape_table.begin(); it != my_data->blendshape_table.end(); ++it) if (it -> second == blnd_nr) break;

		  session_file << blnd_nr + 1 << "," << it -> first << ",";
	              
	          for (int col_nr = 0; col_nr < 9; ++col_nr) 
	                    session_file << my_data -> full_account.at<double>(GenNr_counter * (2 + NumberOfBlendshapes) + 2 + blnd_nr, col_nr) << ",";

		  session_file << my_data -> full_account.at<double>(GenNr_counter * (2 + NumberOfBlendshapes) + 2 + blnd_nr, 9) << std::endl;

      }




      session_file.close();


}    


    
