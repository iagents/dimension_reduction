/**
 * An implementation of the LDA. 
 *
 */

#ifndef _LinearDiscriminantAnalysis_H_
#define _LinearDiscriminantAnalysis_H_

#include <iostream>
#include <assert.h>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/eigen.hpp>

using namespace cv;
using namespace Eigen;

namespace dimension_reduction
{
  using namespace std;
  using namespace cv;
  
  class LDA {
  private:
    Mat Data;
    Mat vecs;
    Mat vals;
    Mat M;	
    Mat weights;
    Mat red_vals;
    
    int num_classes;

  public:
    LDA(Mat Data, Mat labels, int no_classes);
    ~LDA(){}
    //this calculates the projection vectors W
    void init(Mat Data, Mat labels, int no_classes);
    void getEigenValsVecs();
    void sortEigenVecs();
    cv::Mat getWeights();
    //this calculates the projected vector Y = X * W
    cv::Mat project(cv::Mat, cv::Mat);
  };	
}
#endif
