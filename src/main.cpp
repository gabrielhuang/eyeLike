#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>
#include <iostream>
#include <queue>
#include <cmath>

#include "constants.h"
#include "findEyeCenter.h"
#include "findEyeCorner.h"

using namespace std;

void findEyes(cv::Mat frame_gray, cv::Rect face);

template<typename T>
typename T to_range(T x, T a, T b)
{
	if (x < a) return a;
	else if (x > b) return b;
	else return x;
}

// TODO
// account for frame interval
struct Detect
{
	cv::KalmanFilter		kf;
	cv::CascadeClassifier*	classifier;
	cv::Rect				current_face;
	cv::Rect				filtered_face;
	unsigned				max_w;
	unsigned				max_h;
	bool					detected;
	float					d_pos, d_size; // measurement variance
	float					h_pos, h_size; // hysterisis
	bool					use_hysterisis;
	bool					use_filter;
	unsigned				no_face;
	unsigned				max_no_face;

	Detect(cv::CascadeClassifier* classifier)
		: kf(4, 4, 0)
		, max_w(150)
		, max_h(150)
		, classifier(classifier)
		, d_pos(3.f)
		, d_size(30.f)
		, h_pos(7.f)
		, h_size(9.f)
		, use_hysterisis(true)
		, use_filter(true)
		, max_no_face(45) // frames
		, no_face(45)
	{
		kf.transitionMatrix = *(cv::Mat_<float>(4, 4) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);

		kf.statePre.at<float>(0) = 100;
		kf.statePre.at<float>(1) = 100;
		kf.statePre.at<float>(2) = 100;
		kf.statePre.at<float>(3) = 100;
		cv::setIdentity(kf.measurementMatrix);
		cv::setIdentity(kf.processNoiseCov, cv::Scalar::all(1));
		kf.measurementNoiseCov = *(cv::Mat_<float>(4, 4) << 
			d_pos, 0, 0, 0, 0, d_pos, 0, 0, 0, 0, d_size, 0, 0, 0, 0, d_size); 

		cv::setIdentity(kf.errorCovPost, cv::Scalar::all(1.));
	}

	bool operator()(const cv::Mat& frame, cv::Mat& debugImage = cv::Mat())
	{
		// for debug only
		{
			char key = cv::waitKey(5);
			float log_delta = 1.2f;
			if (key == 'P') d_pos *= log_delta;
			else if (key == 'p') d_pos /= log_delta;
			else if (key == 'S') d_size *= log_delta;
			else if (key == 's') d_size /= log_delta;
			kf.measurementNoiseCov = *(cv::Mat_<float>(4, 4) <<
				d_pos, 0, 0, 0, 0, d_pos, 0, 0, 0, 0, d_size, 0, 0, 0, 0, d_size);
			cout << "dpos = " << d_pos << " dsize = " << d_size << endl;
		}

		std::vector<cv::Mat> rgbChannels(3);
		cv::split(frame, rgbChannels);
		cv::Mat frame_gray = rgbChannels[2];
		std::vector<cv::Rect> faces;
		classifier->detectMultiScale(frame_gray, faces, 1.1, 2, 
			0 | CV_HAAR_SCALE_IMAGE | CV_HAAR_FIND_BIGGEST_OBJECT, cv::Size(max_w, max_h));
		
		if (!faces.empty())
		{
			current_face = faces.front();
			
			cv::Mat_<float> m(4, 1); // measurement
			m(0) = current_face.x + float(current_face.width) / 2.f;
			m(1) = current_face.y + float(current_face.height) / 2.f;
			m(2) = current_face.width;
			m(3) = current_face.height;
			kf.predict();
			cv::Mat_<float> corrected = kf.correct(m);

			cv::Rect tmp_face;
			tmp_face.width = to_range<float>(corrected(2), 20, frame_gray.cols - 1);
			tmp_face.height = to_range<float>(corrected(3), 20, frame_gray.rows - 1);
			tmp_face.x = to_range<float>(corrected(0) - corrected(2) / 2.f, 0, frame_gray.cols - filtered_face.width - 1);
			tmp_face.y = to_range<float>(corrected(1) - corrected(3) / 2.f, 0, frame_gray.rows - filtered_face.height - 1);

			if (use_hysterisis)
			{
				float h_p = max(abs(tmp_face.x - filtered_face.x), abs(tmp_face.y - filtered_face.y));
				float h_s = max(abs(tmp_face.width - filtered_face.width), abs(tmp_face.height - filtered_face.height));
				if (h_p > h_pos || h_s > h_size)
				{
					//debug only
					{
						cout << "Moved" << endl;
					}
					filtered_face = tmp_face;
				}
			}
			else
			{
				filtered_face = tmp_face;
			}

			cv::rectangle(debugImage, current_face, cv::Scalar(255, 0, 0), 1);

			no_face = 0;
			detected = true;
		}
		else // not detected
		{
			if (no_face < max_no_face)
			{
				cv::Mat_<float> corrected = kf.predict();
				filtered_face.width = to_range<float>(corrected(2), 20, frame_gray.cols - 1);
				filtered_face.height = to_range<float>(corrected(3), 20, frame_gray.rows - 1);
				filtered_face.x = to_range<float>(corrected(0) - corrected(2) / 2.f, 0, frame_gray.cols - filtered_face.width - 1);
				filtered_face.y = to_range<float>(corrected(1) - corrected(3) / 2.f, 0, frame_gray.rows - filtered_face.height - 1);
				detected = true;
				++no_face;
			}
			else
			{
				detected = false;
			}
		}

		if (detected)
		{
			cv::rectangle(debugImage, filtered_face, cv::Scalar(0, 255, 255), 2);
			cv::circle(debugImage, 
				cv::Point(filtered_face.x + filtered_face.width / 2, filtered_face.y + filtered_face.height / 2),
				3,
				cv::Scalar(0, 255, 0), -1);
			findEyes(frame_gray, filtered_face);
		}

		return detected;
	}
};


int main(int argc, const char** argv) 
{
	std::string face_cascade_name = "../../res/haarcascade_frontalface_alt.xml";
	std::string main_window_name = "Capture - Face detection";
	std::string face_window_name = "Capture - Face";
	cv::RNG rng(12345);
	cv::Mat skinCrCbHist = cv::Mat::zeros(cv::Size(256, 256), CV_8UC1);
	
	try
	{
		cv::CascadeClassifier face_cascade;
		if (!face_cascade.load(face_cascade_name))
		{
			throw std::runtime_error("Cannot find face cascade" + face_cascade_name);
		};

		createCornerKernels();
		cv::ellipse(skinCrCbHist, cv::Point(113, 155.6), cv::Size(23.4, 15.2),
			43.0, 0.0, 360.0, cv::Scalar(255, 255, 255), -1);

		// Read the video stream
		cv::VideoCapture capture(0);
		if (!capture.isOpened())
		{
			throw std::runtime_error("Open VideoCapture");
		}

		cv::Mat frame, debugImage;
		char key = 0;
		Detect detect(&face_cascade);
		while ((key = cv::waitKey(10)) != 27) // ESC
		{
			capture >> frame;
			if (frame.empty())
			{
				throw std::runtime_error("Empty frame");
			}
			cv::flip(frame, frame, 1);
			debugImage = frame.clone();
			detect(frame, debugImage);

			cv::imshow("Debug", debugImage);
		}
	}
	catch (const std::runtime_error& e)
	{
		std::cout << "Exception : " << e.what() << std::endl;
		std::cin.get();
	}
	
	return 0;
}

void findEyes(cv::Mat frame_gray, cv::Rect face) 
{
	cv::Mat faceROI = frame_gray(face);
	cv::Mat debugFace = faceROI;

	if (kSmoothFaceImage) {
		double sigma = kSmoothFaceFactor * face.width;
		GaussianBlur(faceROI, faceROI, cv::Size(0, 0), sigma);
	}
	//-- Find eye regions and draw them
	int eye_region_width = face.width * (kEyePercentWidth / 100.0);
	int eye_region_height = face.width * (kEyePercentHeight / 100.0);
	int eye_region_top = face.height * (kEyePercentTop / 100.0);
	cv::Rect leftEyeRegion(face.width*(kEyePercentSide / 100.0),
		eye_region_top, eye_region_width, eye_region_height);
	cv::Rect rightEyeRegion(face.width - eye_region_width - face.width*(kEyePercentSide / 100.0),
		eye_region_top, eye_region_width, eye_region_height);

	//-- Find Eye Centers
	cv::Point leftPupil = findEyeCenter(faceROI, leftEyeRegion, "Left Eye");
	cv::Point rightPupil = findEyeCenter(faceROI, rightEyeRegion, "Right Eye");
	// get corner regions
	cv::Rect leftRightCornerRegion(leftEyeRegion);
	leftRightCornerRegion.width -= leftPupil.x;
	leftRightCornerRegion.x += leftPupil.x;
	leftRightCornerRegion.height /= 2;
	leftRightCornerRegion.y += leftRightCornerRegion.height / 2;
	cv::Rect leftLeftCornerRegion(leftEyeRegion);
	leftLeftCornerRegion.width = leftPupil.x;
	leftLeftCornerRegion.height /= 2;
	leftLeftCornerRegion.y += leftLeftCornerRegion.height / 2;
	cv::Rect rightLeftCornerRegion(rightEyeRegion);
	rightLeftCornerRegion.width = rightPupil.x;
	rightLeftCornerRegion.height /= 2;
	rightLeftCornerRegion.y += rightLeftCornerRegion.height / 2;
	cv::Rect rightRightCornerRegion(rightEyeRegion);
	rightRightCornerRegion.width -= rightPupil.x;
	rightRightCornerRegion.x += rightPupil.x;
	rightRightCornerRegion.height /= 2;
	rightRightCornerRegion.y += rightRightCornerRegion.height / 2;
	rectangle(debugFace, leftRightCornerRegion, 200);
	rectangle(debugFace, leftLeftCornerRegion, 200);
	rectangle(debugFace, rightLeftCornerRegion, 200);
	rectangle(debugFace, rightRightCornerRegion, 200);
	// change eye centers to face coordinates
	rightPupil.x += rightEyeRegion.x;
	rightPupil.y += rightEyeRegion.y;
	leftPupil.x += leftEyeRegion.x;
	leftPupil.y += leftEyeRegion.y;
	// draw eye centers
	circle(debugFace, rightPupil, 3, 1234);
	circle(debugFace, leftPupil, 3, 1234);

	//-- Find Eye Corners
	if (kEnableEyeCorner) 
	{
		cv::Point2f leftRightCorner = findEyeCorner(faceROI(leftRightCornerRegion), true, false);
		leftRightCorner.x += leftRightCornerRegion.x;
		leftRightCorner.y += leftRightCornerRegion.y;
		cv::Point2f leftLeftCorner = findEyeCorner(faceROI(leftLeftCornerRegion), true, true);
		leftLeftCorner.x += leftLeftCornerRegion.x;
		leftLeftCorner.y += leftLeftCornerRegion.y;
		cv::Point2f rightLeftCorner = findEyeCorner(faceROI(rightLeftCornerRegion), false, true);
		rightLeftCorner.x += rightLeftCornerRegion.x;
		rightLeftCorner.y += rightLeftCornerRegion.y;
		cv::Point2f rightRightCorner = findEyeCorner(faceROI(rightRightCornerRegion), false, false);
		rightRightCorner.x += rightRightCornerRegion.x;
		rightRightCorner.y += rightRightCornerRegion.y;
		circle(faceROI, leftRightCorner, 3, 200);
		circle(faceROI, leftLeftCorner, 3, 200);
		circle(faceROI, rightLeftCorner, 3, 200);
		circle(faceROI, rightRightCorner, 3, 200);
	}

	cv::imshow("Face", faceROI);
}


cv::Mat findSkin(cv::Mat& frame, const cv::Mat& skinCrCbHist)
{
	cv::Mat input;
	cv::Mat output = cv::Mat(frame.rows, frame.cols, CV_8U);

	cvtColor(frame, input, CV_BGR2YCrCb);

	for (int y = 0; y < input.rows; ++y) {
		const cv::Vec3b *Mr = input.ptr<cv::Vec3b>(y);
		//    uchar *Or = output.ptr<uchar>(y);
		cv::Vec3b *Or = frame.ptr<cv::Vec3b>(y);
		for (int x = 0; x < input.cols; ++x) {
			cv::Vec3b ycrcb = Mr[x];
			//      Or[x] = (skinCrCbHist.at<uchar>(ycrcb[1], ycrcb[2]) > 0) ? 255 : 0;
			if (skinCrCbHist.at<uchar>(ycrcb[1], ycrcb[2]) == 0) {
				Or[x] = cv::Vec3b(0, 0, 0);
			}
		}
	}
	return output;
}

