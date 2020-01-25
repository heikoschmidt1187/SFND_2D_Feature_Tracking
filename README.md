# SFND 2D Feature Tracking

<img src="images/keypoints.png" width="820" height="248" />

The idea of the camera course is to build a collision detection system - that's the overall goal for the Final Project. As a preparation for this, you will now build the feature tracking part and test various detector / descriptor combinations to see which ones perform best. This mid-term project consists of four parts:

* First, you will focus on loading images, setting up data structures and putting everything into a ring buffer to optimize memory load.
* Then, you will integrate several keypoint detectors such as HARRIS, FAST, BRISK and SIFT and compare them with regard to number of keypoints and speed.
* In the next part, you will then focus on descriptor extraction and matching using brute force and also the FLANN approach we discussed in the previous lesson.
* In the last part, once the code framework is complete, you will test the various algorithms in different combinations and compare them with regard to some performance measures.

See the classroom instruction and code comments for more details on each of these parts. Once you are finished with this project, the keypoint matching part will be set up and you can proceed to the next lesson, where the focus is on integrating Lidar points and on object detection using deep-learning.

## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./2D_feature_tracking`.

*****

## Writeup

In the following sections will explain how all the [Rubric](https://review.udacity.com/#!/rubrics/2549/view) points have been solved.

### TASK MP.0 - Mid-Term Report
You're reading it! ;-)

### TASK MP.1 - Data Buffer Optimization
The data buffer has been optimized to just hold the last n data elements for processing. The number n can be configured by setting the variable *dataBufferSize* accordingly. You can find the implementation in MidTermProject_Camera_Student.cpp:

```c++
//// STUDENT ASSIGNMENT
//// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize

// push image into data frame buffer
DataFrame frame;
frame.cameraImg = imgGray;

// depending on if the maximum data buffer size is reached, remove the oldest image
if(dataBuffer.size() >= dataBufferSize)
    dataBuffer.erase(dataBuffer.begin());

dataBuffer.push_back(frame);

//// EOF STUDENT ASSIGNMENT
```

The std::vector is filled up until it's size reaches the configured threshold. From that, before adding any new elements, the first and oldest element is removed. This handling is called ring buffer.

### TASK MP.2 - Keypoint Detection
The keypoint detection variants can be selected by setting the string variable *detectorType* to one of the following values:

* SHITOMASI - Shi-Tomasi "good features to track"
* HARRIS - Harris Corner Detector
* FAST - **F**eatures From **A**ccelerated **S**egment **T**est Detector
* BRISK - **B**inary **R**obust **I**nvariant **S**calable **K**eypoints Detector
* ORB - **O**riented FAST and **R**otated **B**RIEF Detector
* AKAZE
* SIFT - **S**cale **I**nvariant **F**eature **T**ransform Detector

In MidTermProject_Camera_Student.cpp, the selected variant leads to differen function calls that handle the detection:

```c++
//// STUDENT ASSIGNMENT
//// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detectorType
//// -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT

if (detectorType.compare("SHITOMASI") == 0) {
    detKeypointsShiTomasi(keypoints, imgGray, false);
} else if (detectorType.compare("HARRIS") == 0) {
    detKeypointsHarris(keypoints, imgGray, false);
} else {
    detKeypointsModern(keypoints, imgGray, detectorType, false);
}
//// EOF STUDENT ASSIGNMENT
```

The implementation can be found in the matching2D_Student.cpp source file. Due to the nature of the implementation, the calls have been divided into the classic algorithms *detKeypointsShiTomasi* and *detKeypointsHarris*, and one common call for the rest - the modern - algorithms *detKeypointsModern*.

The Shi-Tomasi detects corners that are transformed to the OpenCV KeyPoint type and pushed to the resulting vector. The detector parameters have been reused from the lessons:

```c++
// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
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
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    VisualizeKeypoints(img, keypoints, "Shi-Tomasi Corner Detector Results", bVis);
}
```

The same is true for the Harris detector. Instead of taking the detected corners directly, a reduction through a non-maximum suppression in the local neighborhood of detected corners is used before transforming them to keypoints. The parameters again have been reused from the exercises in the lessons:

```c++
// Detect keypoints in image using the traditional Harris detector
void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // Detector parameters
    int blockSize = 2; // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3; // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04; // Harris parameter (see equation for details)

    // Apply corner detection
    double t = (double)cv::getTickCount();

    cv::Mat dst, dst_norm; // result matrices for harris
    dst = cv::Mat::zeros(img.size(), CV_32FC1);

    // detect corners
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);

    // normalize values
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());

    // extract keypoints using non-maximum suppression (NMS)
    double maxOverlap = 0.0;    // max overlap btw. features in %

    // loop over all pixels in the matrix
    for(size_t row = 0; row < dst_norm.rows; ++row) {
        for(size_t col = 0; col < dst_norm.cols; ++col) {

            int response = (int)dst_norm.at<float>(row, col);

            // only responses greater than minResponse will be taken into account
            if(response > minResponse) {
                // create KeyPoint
                cv::KeyPoint newKeyPoint;
                newKeyPoint.pt = cv::Point2f(col, row);
                newKeyPoint.size = 2 * apertureSize;
                newKeyPoint.response = response;

                // perform a non-maximum suppression in local neighborhood around key point
                bool bOverlap = false;

                for(auto it = keypoints.begin(); it != keypoints.end(); ++it) {

                    // get the overlap for the new keypoint
                    double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);

                    // check if overlap is above max
                    if(kptOverlap > maxOverlap) {
                        bOverlap = true;

                        // if overlap is over the maximum, take the keypoint with the
                        // best response
                        if(newKeyPoint.response > (*it).response) {
                            *it = newKeyPoint;
                            break;
                        }
                    }
                }

                // of the overlap flag is not set, add the new point
                if(bOverlap == false)
                    keypoints.push_back(newKeyPoint);
            }
        }
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Harris detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    VisualizeKeypoints(img, keypoints, "Harris Corner Detector Results", bVis);
}
```

The rest of the supported detectors directly return keypoints instead of corners. Due to their similar nature of being used with the OpenCV library, they are combined in one common function call, making use of the cv::Ptr template to incarnate different detector types on one pointer variable:

```c++
// Detect keypoints in image using modern detectors FAST, BRISK, ORB, AKAZE or SIFT
void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis)
{
    // the feature detector pointer
    cv::Ptr<cv::FeatureDetector> detector = nullptr;
    std::string windowName = detectorType + " Keypoint Detector Results";

    if (detectorType.compare("FAST") == 0) {

        // Detector parameters
        int threshold = 30;     // difference between intensity of central px and px of a circle

        // create detector with NMS usage
        cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16;
        detector = cv::FastFeatureDetector::create(threshold, true, type);

    } else if (detectorType.compare("BRISK") == 0) {

        // create detector
        detector = cv::BRISK::create();

    } else if (detectorType.compare("ORB") == 0) {

        // create detector
        detector = cv::ORB::create();

    } else if (detectorType.compare("AKAZE") == 0) {

        // create detector
        detector = cv::AKAZE::create();

    } else if (detectorType.compare("SIFT") == 0) {

        // create detector
        detector = cv::xfeatures2d::SIFT::create();

    } else {
        cout << "Error: Unknown detector type " << detectorType << " configured" << endl;
        return;
    }

    // detect keypoints
    double t = (double)cv::getTickCount();
    detector->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << detectorType << " detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    VisualizeKeypoints(img, keypoints, windowName, bVis);
}
```
To be continued
