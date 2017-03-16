#include "opencv2/opencv.hpp"

using namespace cv;

int main()
{
	VideoCapture cap(0);
	if (!cap.isOpened())
	{
		return -1;
	}

	//Mat edges;
	namedWindow("image capture", 1);

	for (;;)
	{
		Mat frame;
		cap >> frame;
		imwrite("test3.jpg", frame);
		imshow("image capture", frame);
		if (waitKey(30) >= 0) break;
	}
	waitKey(0);
	return 0;
}