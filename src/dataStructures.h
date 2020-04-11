#ifndef dataStructures_h
#define dataStructures_h

#include <vector>
#include <opencv2/core.hpp>


struct DataFrame { // represents the available sensor information at the same time instance
    
    cv::Mat cameraImg; // camera image
    
    std::vector<cv::KeyPoint> keypoints; // 2D keypoints within camera image
    cv::Mat descriptors; // keypoint descriptors
    std::vector<cv::DMatch> kptMatches; // keypoint matches between previous and current frame
};

template < typename T,int MaxLen >
class FixedVector : public std::vector<T> {
public:
    void push_back(const T& value) {
        if (this->size() == MaxLen) {
           this->erase(this->begin());
        }
        std::vector<T>::push_back(value);
    }
};
#endif /* dataStructures_h */
