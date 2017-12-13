#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


using namespace cv;
using namespace std;
int main (int argc, char **argv){
    //std::cout<<CV_VERSION<<std::endl;


    if(argc != 3){
        return 2;
    }

    string origin_path = string(argv[1]);
    string result_path = string(argv[2]);
    //std::cout<<*origin_path<<std::endl;
    cv::Mat origin_mat = imread(origin_path);

    if(origin_mat.empty() || origin_mat.data == nullptr){
        return 3;
    }


    cv::Mat result_mat;
    cv::cvtColor(origin_mat, result_mat, CV_RGB2GRAY);

    imwrite(result_path, result_mat);

    return 0;
}