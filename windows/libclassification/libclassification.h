#ifndef _CAFFE_CLASSIFICATION_DLL_
#define _CAFFE_CLASSIFICATION_DLL_


#include <string>
#include <vector>

#include <opencv2/core/core.hpp>

/* Pair (label, confidence) representing a prediction. */

typedef struct Prediction
{
	int labelid;
	std::string label;
	float score;
}Prediction;

class __declspec(dllexport) Classifier {
public:
	Classifier(const std::string& model_file,
		const std::string& trained_file,
		const std::string& mean_file,
		const std::string& label_file);
	virtual ~Classifier();

	std::vector<Prediction> Classify(const cv::Mat& img, int N = 5);

private:
	void SetMean(const std::string& mean_file);

	std::vector<float> Predict(const cv::Mat& img);

	void WrapInputLayer(std::vector<cv::Mat>* input_channels);

	void Preprocess(const cv::Mat& img,
		std::vector<cv::Mat>* input_channels);

private:
	void* m_net_;
	cv::Size input_geometry_;
	int num_channels_;
	cv::Mat mean_;
	std::vector<std::string> labels_;
};

#endif