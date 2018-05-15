#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

#pragma comment(lib, "opencv_core310.lib");
#pragma comment(lib, "opencv_imgcodecs310.lib");
#pragma comment(lib, "opencv_videoio310.lib");
#pragma comment(lib, "opencv_highgui310.lib");
#pragma comment(lib, "opencv_tracking310.lib");
#pragma comment(lib, "opencv_imgproc310.lib");

using namespace cv;
using namespace std;

int MovieTracking(int argc, char* argv[]){
    if (argc < 2) {
        std::cout << "exe [video]" << std::endl;
        return -1;
    }

    // create Tracker
    cv::Ptr<cv::Tracker> trackerKCF = cv::TrackerKCF::create();
    cv::Ptr<cv::Tracker> trackerTLD = cv::TrackerTLD::create();
    cv::Ptr<cv::Tracker> trackerMEDIANFLOW = cv::TrackerMedianFlow::create();
    cv::Ptr<cv::Tracker> trackerBOOSTING = cv::TrackerBoosting::create();
    cv::Ptr<cv::Tracker> trackerMIL = cv::TrackerMIL::create();

    cv::VideoCapture cap(argv[1]);
    if (!cap.isOpened()) {
        std::cout << "Can't open the video" << std::endl;
        return -1;
    }

    cv::Mat frame;
    cap >> frame;

    // Select tracking target from the frame
    cv::Rect2d roi = cv::selectROI("tracker",frame);
    cv::Rect2d roiTLD = roi;
    cv::Rect2d roiMEDIANFLOW = roi;
    cv::Rect2d roiBOOSTING = roi;
    cv::Rect2d roiMIL = roi;

    cv::Mat target(frame, roi);
    cv::imwrite("target.jpeg", target);
    std::cout << "(x, y, width. height) = (" << roi.x << "," << roi.y << "," << roi.width << "," << roi.height << ")" << std::endl;

    if (roi.width == 0 || roi.height == 0)
        return -1;

    // Initialize Tracker
    trackerKCF->init(frame, roi);
    trackerTLD->init(frame, roiTLD);
    trackerMEDIANFLOW->init(frame, roiMEDIANFLOW);
    trackerBOOSTING->init(frame, roiBOOSTING);
    trackerMIL->init(frame, roiMIL);

    // Set Tracker Color
    cv::Scalar colorkcf = cv::Scalar(0, 255, 0);
    cv::Scalar colortld = cv::Scalar(0, 255, 255);
    cv::Scalar colormedianflow = cv::Scalar(0, 0, 255);
    cv::Scalar colorboosting = cv::Scalar(255, 255, 0);
    cv::Scalar colormit = cv::Scalar(255, 0, 255);

    // Video setting
    double fps = cap.get(CV_CAP_PROP_FPS);
    cv::Size size = cv::Size(cap.get(CV_CAP_PROP_FRAME_WIDTH), cap.get(CV_CAP_PROP_FRAME_HEIGHT));
    const int fourcc = cv::VideoWriter::fourcc('X', 'V', 'I', 'D');
    std::string filename = "output.mp4";
    cv::VideoWriter writer (filename, fourcc, fps, size);

    while(1){
        cap >> frame;
        if(frame.empty())
            break;

        // Update
        trackerKCF->update(frame, roi);
        trackerTLD->update(frame, roiTLD);
        trackerMEDIANFLOW->update(frame, roiMEDIANFLOW);
        trackerBOOSTING->update(frame, roiBOOSTING);
        trackerMIL->update(frame, roiMIL);

        // Surround with box
        cv::rectangle(frame, roi, colorkcf, 1, 1);
        cv::rectangle(frame, roiTLD, colortld, 1, 1);
        cv::rectangle(frame, roiMEDIANFLOW, colormedianflow, 1, 1);
        cv::rectangle(frame, roiBOOSTING, colorboosting, 1, 1);
        cv::rectangle(frame, roiMIL, colormit, 1, 1);

        cv::putText(frame, "- KCF", cv::Point(5, 20), cv::FONT_HERSHEY_SIMPLEX, .5, colorkcf, 1, CV_AA);
        cv::putText(frame, "- TLD", cv::Point(65, 20), cv::FONT_HERSHEY_SIMPLEX, .5, colortld, 1, CV_AA);
        cv::putText(frame, "- MEDIANFLOW", cv::Point(125, 20), cv::FONT_HERSHEY_SIMPLEX, .5, colormedianflow, 1, CV_AA);
        cv::putText(frame, "- BOOSTING", cv::Point(5, 40), cv::FONT_HERSHEY_SIMPLEX, .5, colorboosting, 1, CV_AA);
        cv::putText(frame, "- MIL", cv::Point(115, 40), cv::FONT_HERSHEY_SIMPLEX, .5, colormit, 1, CV_AA);

        cv::imshow("tracker", frame);
        writer << frame;

        int key = cv::waitKey(30);
        if (key == 0x1b)
            break;
    }

    return 0;
}

void MovieRead(){
    VideoCapture video("video.mp4");

    namedWindow("window1");
    while(1){
        Mat frame;
        video >> frame;
        if(frame.empty() || waitKey(30) >= 0 || video.get(CV_CAP_PROP_POS_AVI_RATIO) == 1){
            break;
        }
        imshow("window1",frame);
    }
}

int main(int argc, char* argv[]){
    //MovieRead();
    MovieTracking(argc, argv);
}

