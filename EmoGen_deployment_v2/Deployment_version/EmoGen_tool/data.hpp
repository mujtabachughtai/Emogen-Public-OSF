#ifndef DATA_HPP
#define DATA_HPP

#include <vector>
#include <stdlib.h>
#include <string>
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
# include <map>



class data{

       public:

       std::vector<std::vector<double> > weights_previous_generation;  
       std::vector<std::vector<double> > weights_current_generation; 

       std::vector<int> chosen_rows;

       cv::Mat default_point_cloud = (cv::Mat_<double>(12, 3) << -5.217842282736697, 177.0257362975071, 7.130765661599099,
                                                                 -2.082216018569245, 176.9322759989948, 8.127733707325547,
                                                                  1.214490697492264, 176.7516856478248, 8.006407363988007,
                                                                  4.419770873765433, 176.9097343144138, 6.90815555521209,
                                                                 -2.980542571966907, 169.3703294293009, 8.534801529683818,
                                                                  2.200174334688299, 169.5624411595008, 8.303152316616604,
                                                                 -1.785178563663267, 172.028279552348,  9.215247742345795,
                                                                  1.084627588290445, 172.0598369626787, 9.250439891559395,
                                                                 -0.2654517765085675,170.4770994516044, 10.27503534117955,
                                                                 -0.3800560700757717,165.0406783061944, 9.146746331154276,
                                                                 -0.3270101570943353,168.6324199125165, 10.00619151924616,
                                                                 -0.3468723700447893,177.2295191309015, 9.556766804759832 );

       cv::Mat incoming_point_cloud;

      
        int initialise();
        void prepare_for_next_generation();
        std::map<int,int> left_right_pairs;
        std::map<std::string, int> correctives;
        std::map<std::string, int> blendshape_table;
        std::map<int, std::vector<double>> correctives_to_activation_vector;
        cv::Mat collision_anchor_coordinates;
        cv::Mat collision_anchor_coordinates_teeth;
        cv::Mat lower_lip_coordinates;
     
        
        

        cv::Mat A;
        cv::Mat A_teeth;


        cv::Mat collision_deviations;
        cv::Mat collision_deviations2;         // teeth corrective blendshapes
        cv::Mat collision_deviations_lwr_lip;
        cv::Mat collision_deviations_teeth;
        cv::Mat collision_deviations_teeth2;   // lip corrective blendshapes
        cv::Mat collision_deviations_lwr_lip_teeth;

        std::vector<int> zone1{0,64,65,66,67,69,70,71,72,73,74,75,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,94,95,96};
        std::vector<int> zone2{1,4,55,56,57,58,59,60,61,62,63,68,76,92,93,97,98,99,100,101,102,103,104,105};
        std::vector<int> zone3{2,5,18,19,20,21,26,27,28,29,30,31,32,33,34,35,36,43,44,45,46,47,48,49,50,51,52,53,54};
        std::vector<int> zone4{3,6,7,8,9,10,11,12,13,14,15,16,17,22,23,24,25,37,38,39,40,41,42};
        int session_nr;
      

        std::map<int, std::string> anchor_pair_nr_to_collision_blnd;
        std::map<int, std::string> teeth_anchor_pair_nr_to_collision_blnd;
        std::vector<int> head_motion_blnds;
        std::vector<int> eye_pupil_motion_blnds;
        std::vector<int> eye_lid_motion_blnds;
        std::vector<int> puffs;

        // indexed from 0 (which is the neutral)
        std::vector<int> blnd_subset_happy{31, 32, 35, 36, 39, 40, 45, 46, 47, 48, /* lip depresssor, too much of a sad  51, 52,*/ 53, 54, 66, 67, 68, 69, 73, 74, 75, 76, 77, 78, 79, 80, 83, 84, 93, 94, 1, 41, 42, 49, 50};
        std::vector<int> blnd_subset_sad{6, 51, 52, 64, 65, 66, 67, 70, 71, 73, 74, 83, 84, 93, 94, 1, 4, 68, 69, 7, 8, 39, 40, /* too much happy, lip corner pulls 47, 48,*/ 79, 80, 41, 42};
        std::vector<int> blnd_subset_angry{4, 6, 11, 12, 13, 14, /* seals 19, 20*/ 39, 40, 51, 52, 70, 71, 72, 73, 74, 93, 94, 62, 63, 77, 78, 79, 80, 83, 84, 31, 32, 35, 36, 41, 42, 59, 60, 75, 76, 7, 8, 9, 10, 81, 82};
        std::vector<int> blnd_subset_fearful{1, 5, 6, 39, 40, 41, 42, 64, 65, 66, 67, 68, 69, 70, 71, 75, 76, 93, 94, 4, 11, 12, 13, 14, 43, 44, 79, 80, 83, 84, 35, 36, /* too happy 45, 46 ,*/ 62, 63, 7, 8, 59, 60, 47, 48};

        cv::Mat full_account;

        cv::Mat initialisation;
        std::vector<int> copied_exactly_IDs;
       
       
        data(){
 
            session_nr = 0;

	    weights_previous_generation.resize(10);
	    weights_current_generation.resize(10);

            anchor_pair_nr_to_collision_blnd[0] = "rgtOutClsn";
            anchor_pair_nr_to_collision_blnd[1] = "rgtInrClsn";
            anchor_pair_nr_to_collision_blnd[2] = "lftInrClsn";
            anchor_pair_nr_to_collision_blnd[3] = "lftOutClsn";

            teeth_anchor_pair_nr_to_collision_blnd[0] = "rgtLwrOutLipTeethClsn";
            teeth_anchor_pair_nr_to_collision_blnd[1] = "rgtLwrInrLipTeethClsn";
            teeth_anchor_pair_nr_to_collision_blnd[2] = "lftLwrInrLipTeethClsn";
            teeth_anchor_pair_nr_to_collision_blnd[3] = "lftLwrOutLipTeethClsn";

            // head motion
            for(int blnd_nr=95; blnd_nr < 101; ++blnd_nr) head_motion_blnds.push_back(blnd_nr);
            // eye pupil motion
            for(int blnd_nr=85; blnd_nr < 93; ++blnd_nr) eye_pupil_motion_blnds.push_back(blnd_nr);
            // eye lid motion
            for(int blnd_nr=77; blnd_nr < 81; ++blnd_nr) eye_lid_motion_blnds.push_back(blnd_nr);
            // cheek and lip puffs
            for(int blnd_nr=55; blnd_nr < 59; ++blnd_nr) puffs.push_back(blnd_nr);
             
         

         };
        

};

#endif
