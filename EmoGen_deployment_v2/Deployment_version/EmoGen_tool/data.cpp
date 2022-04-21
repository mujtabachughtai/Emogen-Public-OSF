#include "data.hpp"
#include "utility.hpp"
#include <random>
#include <chrono>

extern int NumberOfBlendshapes;

extern std::string INITIALISATION_FILE;

extern std::mt19937 generator_global;

extern bool random_initialisation, protocol_generated_initialisation;
extern int init_number;
extern bool reinit_after_reset;
extern std::string OUTPUT_DIRECTORY;
extern bool include_head_motion, include_eye_pupil_motion, include_eye_lid_motion;

int data::initialise() {

    // face_order.clear();

    std::cout << "Face Initialisation" << std::endl;

    utility helper = utility();

    if (protocol_generated_initialisation) {

        if (initialisation.rows == 0) {

            std::vector<std::vector<int> *> all_groups;
            all_groups.push_back(&blnd_subset_happy);
            all_groups.push_back(&blnd_subset_sad);
            all_groups.push_back(&blnd_subset_angry);
            all_groups.push_back(&blnd_subset_fearful);

            if (!include_head_motion) {
                for(int blnd_nr = 0; blnd_nr < head_motion_blnds.size(); ++blnd_nr) {
                    for (int group_nr = 0; group_nr < all_groups.size(); ++group_nr) {
                        std::vector<int> &group  = *all_groups[group_nr];
                        std::vector<int>::iterator it  = std::find(group.begin(), group.end(), head_motion_blnds[blnd_nr] );
                        if ( it != group.end() ) group.erase(it);
                    }
                }
            }

            if (!include_eye_pupil_motion) {
                for(int blnd_nr = 0; blnd_nr < eye_pupil_motion_blnds.size(); ++blnd_nr) {
                    for (int group_nr = 0; group_nr < all_groups.size(); ++group_nr) {
                        std::vector<int> &group = *all_groups[group_nr];
                        std::vector<int>::iterator it  = std::find(group.begin(), group.end(), eye_pupil_motion_blnds[blnd_nr] );
                        if ( it != group.end() )  group.erase(it);
                    }
                }
            }

            if (!include_eye_lid_motion) {
                for(int blnd_nr = 0; blnd_nr < eye_lid_motion_blnds.size(); ++blnd_nr) {
                    for (int group_nr = 0; group_nr < all_groups.size(); ++group_nr) {
                        std::vector<int> &group = *all_groups[group_nr];
                        std::vector<int>::iterator it  = std::find(group.begin(), group.end(), eye_lid_motion_blnds[blnd_nr] );
                        if ( it != group.end() ) group.erase(it);
                    }
                }
            }

        }


        if (initialisation.rows == 0 || reinit_after_reset) {
            initialisation = cv::Mat();
            helper.generate_initialisation(initialisation, blnd_subset_happy, blnd_subset_sad, blnd_subset_angry, blnd_subset_fearful, left_right_pairs /*, face_order*/);
            init_number = 1;
        }


    } else {



        if (initialisation.rows == 0) {
            std::cout << "Reading .csv file... " + INITIALISATION_FILE << std::endl;
            cv::Mat initialisation_data_stream = helper.read_csv(INITIALISATION_FILE);

            initialisation = initialisation_data_stream.reshape(0, NumberOfBlendshapes);


            if (initialisation.cols % 10 != 0) {
                std::cout << "Error: incorrect face initialisation file. Exiting.." << std::endl;
                return 1;
            }

            transpose(initialisation, initialisation);
        }

        std::uniform_int_distribution<int> distribution(1, initialisation.rows / 10);
        std::vector<int> init_choices;
        for (int i = 0; i < (initialisation.rows / 10); ++i) init_choices.push_back(i+1);
        if (!random_initialisation && find (init_choices.begin(), init_choices.end(), init_number) == init_choices.end() ) {

            std::cout << "Error: selected initialisation set number " << init_number << " does not exist in the initialisation file. Exiting.." << std::endl;
            return 1;
        } else if (random_initialisation) init_number = distribution(generator_global);



    }


    for (int face_nr = 0; face_nr < 10; ++face_nr){


        if (weights_previous_generation[face_nr].size() != NumberOfBlendshapes) weights_previous_generation[face_nr].resize(NumberOfBlendshapes);
        memset(&weights_previous_generation[face_nr][0], 0, weights_previous_generation[face_nr].size() * sizeof(double));

        if (weights_current_generation[face_nr].size() != NumberOfBlendshapes) weights_current_generation[face_nr].resize(NumberOfBlendshapes);
        memset(&weights_current_generation[face_nr][0], 0, weights_current_generation[face_nr].size() * sizeof(double));


        //std::cout << "Initializing from .csv... setting previous and current weights to the same values.." << std::endl;
        double* data = initialisation.ptr<double>((init_number - 1) * 10 + face_nr,0);

        memcpy(&weights_previous_generation[face_nr][0],
               data,
               NumberOfBlendshapes * sizeof(double));

        memcpy(&weights_current_generation[face_nr][0],
               data,
               NumberOfBlendshapes * sizeof(double));

        if (!protocol_generated_initialisation) {

            if (!include_head_motion) {

                for(int blnd_nr = 0; blnd_nr < head_motion_blnds.size(); ++blnd_nr) {

                    if ( weights_current_generation[face_nr][head_motion_blnds[blnd_nr]] != 0.0) {

                        std::cout << "A POSTERIORI HEAD MOTION DISABLING of face in current position nr., ALEX" << face_nr + 1 << std::endl;

                        weights_previous_generation[face_nr][head_motion_blnds[blnd_nr]] = 0.0;
                        weights_current_generation[face_nr][head_motion_blnds[blnd_nr]] = 0.0;
                    }

                }
            }


            if (!include_eye_pupil_motion) {

                for(int blnd_nr = 0; blnd_nr < eye_pupil_motion_blnds.size(); ++blnd_nr) {

                    if (weights_current_generation[face_nr][eye_pupil_motion_blnds[blnd_nr]] != 0.0) {

                        weights_previous_generation[face_nr][eye_pupil_motion_blnds[blnd_nr]] = 0.0;
                        weights_current_generation[face_nr][eye_pupil_motion_blnds[blnd_nr]] = 0.0;

                    }
                }
            }


            if (!include_eye_lid_motion) {


                for(int blnd_nr = 0; blnd_nr < eye_lid_motion_blnds.size(); ++blnd_nr) {

                    if (weights_current_generation[face_nr][eye_lid_motion_blnds[blnd_nr]] != 0.0) {

                        weights_previous_generation[face_nr][eye_lid_motion_blnds[blnd_nr]] = 0.0;
                        weights_current_generation[face_nr][eye_lid_motion_blnds[blnd_nr]] = 0.0;

                    }

                }
            }

        }



    }
    session_nr++;

    return 0;
};


void data::prepare_for_next_generation(){

    for (int face_nr = 0; face_nr < 10; ++face_nr)
        memcpy(&weights_previous_generation[face_nr][0], &weights_current_generation[face_nr][0],  NumberOfBlendshapes * sizeof(double));

};
