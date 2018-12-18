#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include <opencv2/core/ocl.hpp>
#include <sstream>
#include <string>
#include <vector>
#include <opencv2/tracking.hpp>


using namespace std;
using namespace cv;

//our sensitivity value to be used in the absdiff() function
const static int SENSITIVITY_VALUE = 20;
//size of blur used to smooth the intensity image output from absdiff() function
const static int BLUR_SIZE = 10;
//we'll have just one object to search for
//and keep track of its position.
int theObject[2] = { 0,0 };
//bounding rectangle of the object, we will use the center of this as its position.
Rect objectBoundingRectangle = Rect(0, 0, 0, 0);


//int to string helper function
string intToString(int number) {

	//this function has a number input and string output
	std::stringstream ss;
	ss << number;
	return ss.str();
}
int hMin = 0;
int hMax = 256;
int sMin = 0;
int sMax = 256;
int vMin = 0;
int vMax = 256;
const string windowName = "Original Image";
const string windowName1 = "HSV Image";
const string windowName2 = "Thresholded Image";
const string windowName3 = "After Morphological Operations";
const string trackbarWindowName = "Trackbars";
const int MAX_NUM_OBJECTS = 50;
//minimum and maximum object area
///THESE CONSTANTS ARE SUBJECT TO CHANEG. WRITE A FUNCTION TO DETERMINE THE RESOLUTION OF THE VIDEO
///THE PROGRAM SHOULD BE MADE TO ADJECT WHAT PIXEL AREAS TO IGNORE BASED ON THE RESOLUTION
const int FRAME_HEIGHT = 720;
const int FRAME_WIDTH = 1280;
const int MIN_OBJECT_AREA = 20 * 20;
const int MAX_OBJECT_AREA = FRAME_HEIGHT * FRAME_WIDTH / 1.5;

void on_trackbar(int, void*)
{//This function gets called whenever a
 // trackbar position is changed
}
void createTrackbars() {
	//create window for trackbars
	namedWindow(trackbarWindowName, 0);
	//create memory to store trackbar name on window
	char TrackbarName[50];
	sprintf_s(TrackbarName, "H_MIN", hMin);
	sprintf_s(TrackbarName, "H_MAX", hMax);
	sprintf_s(TrackbarName, "S_MIN", sMin);
	sprintf_s(TrackbarName, "S_MAX", sMax);
	sprintf_s(TrackbarName, "V_MIN", vMin);
	sprintf_s(TrackbarName, "V_MAX", vMax);
	//create trackbars and insert them into window
	//3 parameters are: the address of the variable that is changing when the trackbar is moved(eg.H_LOW),
	//the max value the trackbar can move (eg. H_HIGH), 
	//and the function that is called whenever the trackbar is moved(eg. on_trackbar)
	//                                  ---->    ---->     ---->      
	createTrackbar("H_MIN", trackbarWindowName, &hMin, 256, on_trackbar);
	createTrackbar("H_MAX", trackbarWindowName, &hMax, 256, on_trackbar);
	createTrackbar("S_MIN", trackbarWindowName, &sMin, 256, on_trackbar);
	createTrackbar("S_MAX", trackbarWindowName, &sMax, 256, on_trackbar);
	createTrackbar("V_MIN", trackbarWindowName, &vMin, 256, on_trackbar);
	createTrackbar("V_MAX", trackbarWindowName, &vMax, 256, on_trackbar);
}
void morphOps(Mat &thresh) {
	//create structuring element that will be used to "dilate" and "erode" image.
	//the element chosen here is a 3px by 3px rectangle

	Mat erodeElement = getStructuringElement(MORPH_RECT, Size(3, 3));
	//dilate with larger element so make sure object is nicely visible
	Mat dilateElement = getStructuringElement(MORPH_RECT, Size(8, 8));

	erode(thresh, thresh, erodeElement);
	erode(thresh, thresh, erodeElement);

	dilate(thresh, thresh, dilateElement);
	dilate(thresh, thresh, dilateElement);
}
void drawObject(int x, int y, Mat &frame) {

	//use some of the openCV drawing functions to draw crosshairs
	//on your tracked image!

	//UPDATE:JUNE 18TH, 2013
	//added 'if' and 'else' statements to prevent
	//memory errors from writing off the screen (ie. (-25,-25) is not within the window!)

	circle(frame, Point(x, y), 20, Scalar(0, 255, 0), 2);
	if (y - 25>0)
		line(frame, Point(x, y), Point(x, y - 25), Scalar(0, 255, 0), 2);
	else line(frame, Point(x, y), Point(x, 0), Scalar(0, 255, 0), 2);
	if (y + 25<FRAME_HEIGHT)
		line(frame, Point(x, y), Point(x, y + 25), Scalar(0, 255, 0), 2);
	else line(frame, Point(x, y), Point(x, FRAME_HEIGHT), Scalar(0, 255, 0), 2);
	if (x - 25>0)
		line(frame, Point(x, y), Point(x - 25, y), Scalar(0, 255, 0), 2);
	else line(frame, Point(x, y), Point(0, y), Scalar(0, 255, 0), 2);
	if (x + 25<FRAME_WIDTH)
		line(frame, Point(x, y), Point(x + 25, y), Scalar(0, 255, 0), 2);
	else line(frame, Point(x, y), Point(FRAME_WIDTH, y), Scalar(0, 255, 0), 2);

	putText(frame, intToString(x) + "," + intToString(y), Point(x, y + 30), 1, 1, Scalar(0, 255, 0), 2);

}
int trackFilteredObject(int &x, int &y, Mat threshold, Mat &cameraFeed) {

	Mat temp;
	threshold.copyTo(temp);
	//these two vectors needed for output of findContours
	vector< vector<Point> > contours;
	vector<Vec4i> hierarchy;
	//find contours of filtered image using openCV findContours function
	findContours(temp, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	//use moments method to find our filtered object
	double refArea = 0;
	bool objectFound = false;
	if (hierarchy.size() > 0) {
		int numObjects = hierarchy.size();
		//if number of objects greater than MAX_NUM_OBJECTS we have a noisy filter
		if (numObjects<MAX_NUM_OBJECTS) {
			for (int index = 0; index >= 0; index = hierarchy[index][0]) {

				Moments moment = moments((cv::Mat)contours[index]);
				double area = moment.m00;

				//if the area is less than 20 px by 20px then it is probably just noise
				//if the area is the same as the 3/2 of the image size, probably just a bad filter
				//we only want the object with the largest area so we safe a reference area each
				//iteration and compare it to the area in the next iteration.
				if (area>MIN_OBJECT_AREA && area<MAX_OBJECT_AREA && area>refArea) {
					x = moment.m10 / area;
					y = moment.m01 / area;
					objectFound = true;
					refArea = area;
				}
				else objectFound = false;
			}
			//let user know you found an object
			if (objectFound == true) {
				putText(cameraFeed, "Tracking Object", Point(0, 50), 2, 1, Scalar(0, 255, 0), 2);
				//draw object location on screen
				drawObject(x, y, cameraFeed);
			}
		}
		else putText(cameraFeed, "TOO MUCH NOISE! ADJUST FILTER", Point(0, 50), 1, 2, Scalar(0, 0, 255), 2);
	}
	return x, y;
}
void processImage(int frames) {
	Mat  dullFrame, frame;
	VideoCapture pCapture;

	//getting the number of frames
	pCapture.open("videos/Swing1.1.mp4");

	for (int x = 0; x < frames; x++) {
		pCapture.read(frame);

		cvtColor(frame, dullFrame, COLOR_BGR2HSV);

		for (int r = 0; r < dullFrame.rows; r++) {

			for (int c = 0; c < dullFrame.cols; c++) {
				cout << r << "," << c << endl;
				//Checking if value and saturation range is too low for specified colours
				if (dullFrame.at<Vec3b>(r, c)[1] > 200 && dullFrame.at<Vec3b>(r, c)[2] < 55) {
					dullFrame.at<Vec3b>(r, c) = dullFrame.at<Vec3b>(r, c) * 0.001f;
				}

			}
		}
	}
	

	//Looking for very saturated values

	namedWindow("ImageMod", WINDOW_NORMAL);
	namedWindow("ImageMod", CV_WINDOW_AUTOSIZE);
	imshow("ImageMod", dullFrame);
}
int TrackObject() {
	///these two can be toggled on for demonstration. but can be removed and have their values hard coded
	bool trackObjects = true;
	bool useMorphOps = true;
	//set up the matrices that we will need
	Mat frame1;
	//HSV image
	Mat hsvImage1;
	//resulting difference image
	Mat threshold;
	Mat HSVImage;

	//variable to open the video
	VideoCapture capture;
	int x,n, y = 0;
	
	//getting the number of frames
	capture.open("videos/Swing1.1.mp4");
	int fCount = capture.get(CAP_PROP_FRAME_COUNT);
	capture.release();

	//GOing through the image and toning down the values that I want
	processImage(fCount);
	//initializing matrix with values that can be used later for debugging.
	vector <vector<int> > positionA(fCount, vector<int>(99999, 99999));

	//Getting the length of the video so an array can be made to store the position on each frame
	//TYRING TO GET ARRAYS TO WORK
	
	//remove this to loop. hard code while (1)
	int k = 1;
	while (k) {

		//we can loop the video by re-opening the capture every time the video reaches its last frame

		capture.open("videos/Swing1.1.mp4");

		if (!capture.isOpened()) {
			cout << "ERROR ACQUIRING VIDEO FEED\n";
			getchar();
			return -1;
		}
		
		n = 0;
		//check if the video has reach its last frame.
		//we add '-1' because we are reading two frames from the video at a time.
		//if this is not included, we get a memory error!
		while (capture.get(CV_CAP_PROP_POS_FRAMES)<capture.get(CV_CAP_PROP_FRAME_COUNT) - 1) {
			
			//read first frame
			capture.read(frame1);
			//convert to HSV
			cvtColor(frame1, hsvImage1, COLOR_BGR2HSV);
			cvtColor(frame1, HSVImage, COLOR_BGR2HSV);

			//filter by threshold (user defined
			createTrackbars();
			///For now I will hard code the values. Just so I don't have to calibrate every time
			Mat threshold;
			inRange(hsvImage1, Scalar(hMin, sMin, vMin), Scalar(hMax, sMax, vMax), threshold);
			//inRange(HSVImage, Scalar(165, 121, 153), Scalar(256, 256, 256), threshold);
			
			//Eroding and Dialating to remove noise
			///We can remove the IF statement later but just in case I have to demonstrate the difference I will keep this
			if (useMorphOps)
				morphOps(threshold);

			if (trackObjects) {
				//pass in thresholded frame to our object tracking function
				//this function will return the x and y coordinates of the
				//trackFilteredObject(x value, y, threshold, cameraFeed);
				trackFilteredObject(x, y, threshold, frame1);
			}

			namedWindow("HSV Image", WINDOW_NORMAL);
			namedWindow("HSV Image", CV_WINDOW_AUTOSIZE);
			imshow("HSV Image", HSVImage);

			namedWindow("Threshold Image", WINDOW_NORMAL);
			namedWindow("Threshold Image", CV_WINDOW_AUTOSIZE);
			imshow("Threshold Image", threshold);

			namedWindow("Frame1", WINDOW_NORMAL);
			namedWindow("Frame1", CV_WINDOW_AUTOSIZE);
			imshow("Frame1", frame1);
			
			waitKey(10);

			positionA[n][0] = x;
			positionA[n][1] = y;

			n++;
		}
		//k = 0;
		//release the capture before re-opening and looping again.
		capture.release();
	}
	//printing the value of the ball's coordinates
	for (auto vec : positionA)
	{
		std::cout << vec[0] << " , " << vec[1] << endl;
	}

	return 0;
}
int main() {
	TrackObject();
	return 0;
}