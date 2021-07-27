
#include "ceres/ceres.h"
#include <cmath>

// opencv
# include <opencv2/core/core.hpp>
# include "opencv/cv.h"
# include "opencv2/opencv.hpp"


class CorrectiveWeightsAnalytic: public ceres::SizedCostFunction < 2, 8 > {
    
  public:

  CorrectiveWeightsAnalytic(cv::Mat &Arow, double &b_init, bool lip_constraint) 
  {
	  Arow.copyTo(A);
	  b = b_init;
	  typeOfconstraint = lip_constraint;
  };
  
  virtual ~CorrectiveWeightsAnalytic() {};
  
  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {
    

       double sum = 0.0;
       cv::Mat collision_weights(1,8, CV_64F);
       collision_weights.setTo(0.0);
       for (int k = 0; k < 8; ++k)  sum = sum + A.at<double>(k) * parameters[0][k];  
       for (int k = 0; k < 8; ++k) collision_weights.at<double>(k) = parameters[0][k]; 

      
       double norm_collision_weights = cv::norm(collision_weights);

        residuals[0] = dw * (sum - b);
        residuals[1] = regw * norm_collision_weights;

       
	
       if (!jacobians) return true;
      
       if (jacobians[0] != NULL )  {

             if (typeOfconstraint) { // lip constraints

		     for (int k = 0; k < 4 ; ++k)  jacobians[0][k] = dw * A.at<double>(k);
                     for (int k = 4; k < 8 ; ++k)  jacobians[0][k] = 0.0; 
                     for (int k = 8; k < 12 ; ++k) {
                         int k_adj = k - 8;
                         if (norm_collision_weights != 0.0)  jacobians[0][k]  = regw * parameters[0][k_adj] / norm_collision_weights;
                         else jacobians[0][k]  = 0.0;
                     }

                     for (int k = 12; k < 16 ; ++k)  jacobians[0][k]  = 0.0;
                   
             }  else { // teeth constraints

                for (int k = 0; k < 4 ; ++k) jacobians[0][k] =  0.0;
                for (int k = 4; k < 8 ; ++k) jacobians[0][k] = dw * A.at<double>(k);

                for (int k = 8; k < 12 ; ++k)  jacobians[0][k]  = 0.0;
                
                for (int k = 12; k < 16 ; ++k) {
                       int k_adj = k - 8;
                       if (norm_collision_weights != 0.0)  jacobians[0][k]  = regw * parameters[0][k_adj] / norm_collision_weights;
                       else jacobians[0][k]  = 0.0;
                }
            
		 

             }

        }
       
       return true;

    }

    private:

	   cv::Mat A;
	   double b;
	   bool typeOfconstraint;
           double dw = sqrt(0.98);
           double regw = sqrt(0.02);
 

};

