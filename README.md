# SFND 2D Feature Tracking

<img src="images/keypoints.png" width="820" height="248" />

# SFND 2D Feature Tracking

This project is the second in a series for Udacity's Sensor Fusion Nanodegree. The project covers the following key concepts:

- Using ring buffers to avoid memory bloat while processing a sequence of images
- Keypoint detectors such as: Shi-Tomasi, Harris, FAST, BRISK, ORB, AKAZE, and SIFT
- Keypoint descriptor extraction and matching with: FLANN and k-NN

These techniques provide a foundation for the next step: time-to-collision estimation.
## Overview of the workflow
1. Load the images into a ring buffer. 
1. Use OpenCV to apply a variety of keypoint detectors.
    - Shi-Tomasi
    - Harris
    - FAST
    - BRISK
    - ORB
    - AKAZE
    - SIFT (Patent encumbered, https://patents.google.com/patent/US6711293B1/en)
1. Use OpenCV to extract keypoint descriptors.
    - BRISK
    - BRIEF
    - ORB
    - FREAK
    - AKAZE
    - SIFT 
1. Use FLANN and kNN to improve on the brute force matching of keypoint descriptors.
1. Finally, run these algorithms in various combinations to compare performance benchmarks.

It's important to distinguish between the terms of art keypoint **detector** and keypoint **descriptor**. From Udacity's lecture notes:
> - A keypoint (sometimes also interest point or salient point) detector is an algorithm that chooses points from an image based on a local maximum of a function, such as the "cornerness" metric we saw with the Harris detector.
> - A descriptor is a vector of values, which describes the image patch around a keypoint. There are various techniques ranging from comparing raw pixel values to much more sophisticated approaches such as histograms of gradient orientations.

## Dependencies
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* OpenCV >= 4.1
  * On macOS, simply `brew install opencv`
  * If compiled from source, ensure that cmake flag is set `-D OPENCV_ENABLE_NONFREE=ON` for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Building and running the project
```
mkdir build && cd build
cmake ..
make
./2D_feature_tracking
```

-----
# Result 


#### 1. MP.1 Data Buffer Optimization

in dataStructures.h
```c++
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
```
in MidTermProject_Camera_Student.cpp
```c++
const int dataBufferSize = 2; 
FixedVector<DataFrame, dataBufferSize> dataBuffer;
```

#### 2. MP.2 Keypoint Detection

MidTermProject_Camera_Student.cpp
```c++

/* detTime used to store detector time for all images to be used in total AvgTime instead of calculate it manually */

if (detectorType.compare("SHITOMASI") == 0)
{
    detKeypointsShiTomasi(keypoints, imgGray, false, &detTime);
}
else if (detectorType.compare("HARRIS") == 0)
{
    detKeypointsHarris(keypoints, imgGray, false, &detTime);
}
else
    detKeypointsModern(keypoints, imgGray, detectorType, false, &detTime);

```
matching2D_Student.cpp
```c++
void vis(string windowName, cv::Mat &img, std::vector<cv::KeyPoint> &keypoints, cv::Mat &visImg)
{
    cv::namedWindow(windowName, 5);
    //cv::Mat visImage = dst_norm_scaled.clone();
    cv::drawKeypoints(img, keypoints, visImg, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::imshow(windowName, visImg);
    cv::waitKey(0);
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis, double *time)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {
        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    (*time)+=(1000 * t / 1.0);
    //cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    // visualize results
    if (bVis)
    {
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::Mat visImage = img.clone();
        vis(windowName,img,keypoints,visImage);
        
    }
}

void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis, double  *time)
{
    int blockSize = 4;
    double  k = 0.04;
    int ksize = 3;
    int minResponse = 100;
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);

    double t = (double)cv::getTickCount();

    cv::cornerHarris(img, dst, blockSize, ksize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    double maxOverlap = 0.0; 
    for (size_t j = 0; j < dst_norm.rows; j++)
    {
        for (size_t i = 0; i < dst_norm.cols; i++)
        {
            int response = (int)dst_norm.at<float>(j, i);
            if (response > minResponse)
            { 

                cv::KeyPoint newKeyPoint;
                newKeyPoint.pt = cv::Point2f(i, j);
                newKeyPoint.size = 2 * ksize;
                newKeyPoint.response = response;

                bool bOverlap = false;
                for (auto it = keypoints.begin(); it != keypoints.end(); ++it)
                {
                    double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);
                    if (kptOverlap > maxOverlap)
                    {
                        bOverlap = true;
                        if (newKeyPoint.response > (*it).response)
                        {                      
                            *it = newKeyPoint; 
                            break;             
                        }
                    }
                }
                if (!bOverlap)
                {                                    
                    keypoints.push_back(newKeyPoint); 
                }
            }
        } // eof loop over cols
    }     // eof loop over rows
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    (*time)+=(1000 * t / 1.0);

    if(bVis)
    {
        // visualize keypoints
        string windowName = "Harris Corner Detection Results";
        cv::Mat visImage = dst_norm_scaled.clone();
        vis(windowName,dst_norm_scaled,keypoints,visImage);
        
    }

}
void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis, double *time)
{
    if (detectorType.compare("FAST") == 0)
    {
        int threshold = 20;
        cv::Ptr<cv::FastFeatureDetector> detector=cv::FastFeatureDetector::create(threshold,true);
        double t = (double)cv::getTickCount();
        detector->detect(img,keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        (*time)+=(1000 * t / 1.0);

    }
    else if (detectorType.compare("BRISK") == 0)
    {
        cv::Ptr<cv::BRISK> detector=cv::BRISK::create();
        double t = (double)cv::getTickCount();
        detector->detect(img,keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        (*time)+=(1000 * t / 1.0);

    }

    else if (detectorType.compare("SIFT") == 0)
    {
        int nfeatures = 100;
        cv::Ptr<cv::FeatureDetector> detector = cv::xfeatures2d::SIFT::create();
        double t = (double)cv::getTickCount();
        detector->detect(img,keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        (*time)+=(1000 * t / 1.0);

    }
    else if (detectorType.compare("ORB") == 0)
    {
        int nfeatures = 100;
        cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
        double t = (double)cv::getTickCount();
        detector->detect(img,keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        (*time)+=(1000 * t / 1.0);


    }
    else if (detectorType.compare("AKAZE") == 0)
    {
        int nfeatures = 100;
        cv::Ptr<cv::FeatureDetector> detector = cv::AKAZE::create();
        double t = (double)cv::getTickCount();
        detector->detect(img,keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        (*time)+=(1000 * t / 1.0);

    }

    if(bVis)
    {
        // visualize keypoints
        string windowName = "FAST Corner Detection Results";
        cv::Mat visImage = img.clone();
        vis(windowName,img,keypoints,visImage);

    }
}
```
#### 3. MP.3 Keypoint Removal
MidTermProject_Camera_Student.cpp
```c++
// only keep keypoints on the preceding vehicle
bool bFocusOnVehicle = true;
cv::Rect vehicleRect(535, 180, 180, 150);
vector<cv::KeyPoint>::iterator keypoint;
vector<cv::KeyPoint> keypoints_roi;
if (bFocusOnVehicle)
{
    for(keypoint = keypoints.begin(); keypoint != keypoints.end(); ++keypoint)
    {
        if (vehicleRect.contains(keypoint->pt))
        {  
            cv::KeyPoint newKeyPoint;
            newKeyPoint.pt = cv::Point2f(keypoint->pt);
            newKeyPoint.size = 1;
            keypoints_roi.push_back(newKeyPoint);
        }
    }
    keypoints =  keypoints_roi;
    total_ROI_kps += keypoints.size();
}
```

#### 4. MP.4 Keypoint Descriptors

MidTermProject_Camera_Student.cpp
```c++
descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType);
```
matching2D_Student.cpp
```c++
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType, double *time)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.
        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if(descriptorType.compare("BRIEF") == 0)
    {
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    }
    else if (descriptorType.compare("ORB") == 0)
    {
        extractor = cv::ORB::create();
    }
    else if (descriptorType.compare("FREAK") == 0)
    {
        extractor = cv::xfeatures2d::FREAK::create();
    }
    else if (descriptorType.compare("AKAZE") == 0)
    {
        extractor = cv::AKAZE::create();
    }
    else if (descriptorType.compare("SIFT") == 0)
    {
        extractor = cv::xfeatures2d::SIFT::create();
    }
    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    (*time)+=(1000 * t / 1.0);
}
```

#### 6. MP.5 Descriptor Matching And MP.6 Descriptor Distance Ratio


matching2D_Student.cpp
```c++
if (matcherType.compare("MAT_BF") == 0)
{
    int normType = cv::NORM_HAMMING;
    matcher = cv::BFMatcher::create(normType, crossCheck);
}
else if (matcherType.compare("MAT_FLANN") == 0)
{
    if (descSource.type() != CV_32F)
    { // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
        descSource.convertTo(descSource, CV_32F);
        descRef.convertTo(descRef, CV_32F);
    }
    matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
}

// perform matching task
if (selectorType.compare("SEL_NN") == 0)
{ // nearest neighbor (best match)

    matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
}
else if (selectorType.compare("SEL_KNN") == 0)
{ // k nearest neighbors (k=2)

    std::vector<std::vector<cv::DMatch>> knn_matcher;
    matcher->knnMatch(descSource,descRef,knn_matcher,2);
    float distThs = 0.8;
    for(auto it = knn_matcher.begin(); it != knn_matcher.end();++it)
    {
        if((*it)[0].distance <= distThs * (*it)[1].distance)
            matches.push_back((*it)[0]);
    }
}
```

#### 7. MP.7 Performance Evaluation 1
Count the number of keypoints on the preceding vehicle for all 10 images and take note of the distribution of their neighborhood size. Do this for all the detectors you have implemented.

| Detector | Number of detected Keypoints on the preceding vehicle for total of 10 images |
| --- | --- |
| **SHITOMASI** | 1179 |
| **HARRIS** | 176 |
| **FAST** | 2207 |
| **BRISK** | 2762 |
| **ORB** | 1161 |
| **AKAZE** | 1670 |
| **SIFT** | 1386 |

#### 8. MP.8 Performance Evaluation 2

Count the number of matched keypoints for all 10 images using all possible combinations of detectors and descriptors. In the matching step, the BF approach is used with the descriptor distance ratio set to 0.8.


| Detector\Descriptor | BRISK | BRIEF | ORB | FREAK | AKAZE | SIFT |
| --- | --- | --- |--- |--- |--- |--- |
| **SHITOMASI** | 770 |947|910|770|N/A|788|
| **HARRIS** | 147|146 |176|127|N/A|122|
| **FAST** | 1290 |1615|1590|1285|N/A|1307|
| **BRISK** | 1526 |1727|1769|1470|N/A|1364|
| **SIFT** | 591 |705|755|593|N/A|630|
| **ORB** | 564 |558|603|515|N/A|511|
| **AKAZE** | 1211 |1276|1318|1185|N/A|1082|

#### MP.9 Performance Evaluation 3

Log the time it takes for keypoint detection and descriptor extraction

| Detector\Descriptor | BRISK | BRIEF | ORB | FREAK | AKAZE | SIFT |
| --- | --- | --- |--- |--- |--- |--- |
| **SHITOMASI** | 16.867 |19.758|21.57|50.358|N/A| 24.961|
| **HARRIS** | 17.535|17.935 |10.812|51.836| N/A| 29.123|
| **FAST** | 4.967 |6.6411 |8.836|48.045|N/A|23.795|
| **BRISK** | 31.178 |29.534|31.928|59.906|N/A|38.011|
| **SIFT** | 83.8 |81.758|96.213|115.855|N/A|91.175|
| **ORB** | 17.019 |21.632|26.023|58.558|N/A|33.367|
| **AKAZE** | 54.199 |51.824|51.48|80.068|N/A|59.408|

Based on the results above, the top 3 detector/descriptor combinations, that achieve minimal processing time with significant matches are:

DETECTOR/DESCRIPTOR  | NUMBER OF KEYPOINTS | TIME
-------------------- | --------------------| --------
FAST+BRISK           | 1526 keypoints    | 4.96 ms 
FAST+BRIEF             | 1615 keypoints    | 6.641 ms 
FAST+ORB            | 1590 keypoints     | 8.83 ms 