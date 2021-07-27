#ifndef CORRECTIVEWEIGHTSANALYTIC_H
#define CORRECTIVEWEIGHTSANALYTIC_H

#include "ceres/ceres.h"
#include <cmath>

// opencv
# include <opencv2/core/core.hpp>
# include "opencv/cv.h"
# include "opencv2/opencv.hpp"



class CorrectiveWeightsAnalytic: public ceres::SizedCostFunction < 2 ,8> {
    
  public:

  CorrectiveWeightsAnalytic(const double x, const double y) : x_(x), y_(y) {};
  virtual ~CorrectiveWeightsAnalytic() {};
  virtual bool Evaluate(double const* const* parameters,
                           double* residuals,
                           double** jacobians) const;
   private:
   const double x_;
   const double y_;
 

};


#endif
