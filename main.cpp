#include <iostream>
#include <opencv2/opencv.hpp>
#include "prosac.h"

using namespace cv;
using namespace std;


int main(int argc, char** argv)
{
	Mat obj = imread("..\\images\\obj.jpg");   //载入目标图像
	resize(obj, obj, Size(obj.cols / 2, obj.rows / 2));
	Mat scene = imread("..\\images\\scene.jpg"); //载入场景图像

	if (obj.empty() || scene.empty())
	{
		cout << "Can't open the picture!\n";
		return 0;
	}

	vector<KeyPoint> obj_keypoints, scene_keypoints;
	Mat obj_descriptors, scene_descriptors;
	Ptr<ORB> detector = ORB::create();     //采用ORB算法提取特征点
	detector->detectAndCompute(obj, Mat(), obj_keypoints, obj_descriptors);
	detector->detectAndCompute(scene, Mat(), scene_keypoints, scene_descriptors);

	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(NORM_HAMMING); //汉明距离做为相似度度量
	vector<DMatch> matches;
	matcher->match(obj_descriptors, scene_descriptors, matches);

	Mat match_img;
	drawMatches(obj, obj_keypoints, scene, scene_keypoints, matches, match_img);
	//imshow("滤除误匹配前", match_img);

	//保存匹配对序号
	vector<int> queryIdxs(matches.size()), trainIdxs(matches.size());
	for (size_t i = 0; i < matches.size(); i++)
	{
		queryIdxs[i] = matches[i].queryIdx;
		trainIdxs[i] = matches[i].trainIdx;
	}

	Mat H12;   //变换矩阵
	vector<Point2f> points1; KeyPoint::convert(obj_keypoints, points1, queryIdxs);
	vector<Point2f> points2; KeyPoint::convert(scene_keypoints, points2, trainIdxs);
	int ransacReprojThreshold = 5;  //拒绝阈值

	int prosac_T_N = nchoosek((int)points1.size() / 5, 4);
	double sTime = (double)getTickCount();
	H12 = findHomographyProsac(Mat(points1), Mat(points2), CV_PROSAC, prosac_T_N, ransacReprojThreshold);
	double interval = ((double)getTickCount() - sTime) / getTickFrequency();
	cout << "Find homography time: " << interval << "s" << endl;

	vector<char> matchesMask(matches.size(), 0);
	Mat points1t;
	perspectiveTransform(Mat(points1), points1t, H12);

	for (size_t i1 = 0; i1 < points1.size(); i1++)  //保存‘内点’
	{
		if (norm(points2[i1] - points1t.at<Point2f>((int)i1, 0)) <= ransacReprojThreshold) //给内点做标记
		{
			matchesMask[i1] = 1;
		}
	}

	Mat match_img2;   //滤除‘外点’后
	drawMatches(obj, obj_keypoints, scene, scene_keypoints, matches, match_img2, Scalar::all(-1), Scalar::all(-1), matchesMask);

	//画出目标位置
	std::vector<Point2f> obj_corners(4);
	obj_corners[0] = cvPoint(0, 0); obj_corners[1] = cvPoint(obj.cols, 0);
	obj_corners[2] = cvPoint(obj.cols, obj.rows); obj_corners[3] = cvPoint(0, obj.rows);
	std::vector<Point2f> scene_corners(4);
	perspectiveTransform(obj_corners, scene_corners, H12);
	line(match_img2, scene_corners[0] + Point2f(static_cast<float>(obj.cols), 0),
		scene_corners[1] + Point2f(static_cast<float>(obj.cols), 0), Scalar(0, 0, 255), 2);
	line(match_img2, scene_corners[1] + Point2f(static_cast<float>(obj.cols), 0),
		scene_corners[2] + Point2f(static_cast<float>(obj.cols), 0), Scalar(0, 0, 255), 2);
	line(match_img2, scene_corners[2] + Point2f(static_cast<float>(obj.cols), 0),
		scene_corners[3] + Point2f(static_cast<float>(obj.cols), 0), Scalar(0, 0, 255), 2);
	line(match_img2, scene_corners[3] + Point2f(static_cast<float>(obj.cols), 0),
		scene_corners[0] + Point2f(static_cast<float>(obj.cols), 0), Scalar(0, 0, 255), 2);

	//imshow("滤除误匹配后", match_img2);
	//waitKey(0);
	imwrite("img1.png", match_img);
	imwrite("img2.png", match_img2);

	return 0;
}