#include "opencv2/core/core.hpp"
//#include "opencv2/contrib_world.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
	std::ifstream file(filename.c_str(), ifstream::in);
	if (!file) {
		string error_message = "No valid input file was given, please check the given filename.";
		CV_Error(CV_StsBadArg, error_message);
	}
	string line, path, classlabel;
	while (getline(file, line)) {
		stringstream liness(line);
		getline(liness, path, separator);
		getline(liness, classlabel);
		if (!path.empty() && !classlabel.empty()) {
			images.push_back(imread(path, 0));
			labels.push_back(atoi(classlabel.c_str()));
		}
	}
}

int main(int argc, const char *argv[]) {
	
	string fn_haar = "haarcascade_frontalface_alt.xml";
	string fn_csv = "at.txt";
	
	vector<Mat> images;
	vector<int> labels;
	images.push_back(imread("test.jpg",0));
	labels.push_back(2);
	/*try {
		//read_csv(fn_csv, images, labels);
	}
	catch (cv::Exception& e) {
		cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
		exit(1);
	}*/

	images.push_back(imread("test.jpg", 0));
	labels.push_back(1);

	int im_width = images[0].cols;
	int im_height = images[0].rows;
	
	Ptr<face::FaceRecognizer> model = face::createFisherFaceRecognizer();
	model->train(images, labels);

	CascadeClassifier haar_cascade;
	haar_cascade.load(fn_haar);

	Mat img = imread("test2.jpg");
	Mat gray;
	cvtColor(img, gray, CV_BGR2GRAY);
	vector<Rect_<int>> faces;
	haar_cascade.detectMultiScale(gray, faces);

	for (int i = 0; i < faces.size(); i++) {
		
		Rect face_i = faces[i];
	
		Mat face = gray(face_i);
		Mat face_resized;
		cv::resize(face, face_resized, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);

		double pred_confidence = 0.0;
		int prediction;
		model->predict(face_resized,prediction,pred_confidence);
		

		rectangle(img, face_i, CV_RGB(0, 255, 0), 1);

		string box_text = format("Pred = %d Conf = %lf", prediction, pred_confidence);

		int pos_x = std::max(face_i.tl().x - 10, 0);
		int pos_y = std::max(face_i.tl().y - 10, 0);

		putText(img, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 0), 2.0);

		// Show the result:
		imshow("face_recognizer", img);
		printf("Pred = %d Conf = %lf", prediction, pred_confidence);

		char key = (char)waitKey(20);
		
		if (key == 27)
			break;
	}
	waitKey(0);
	return 0;
}