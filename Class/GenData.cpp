

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/ml/ml.hpp>

#include<iostream>
#include<vector>


const int MIN_CONTOUR_AREA = 100;

const int RESIZED_IMAGE_WIDTH = 20;
const int RESIZED_IMAGE_HEIGHT = 30;


int main() {

	cv::Mat TrainingImgN;         
	cv::Mat imgGrayscale;               
	cv::Mat imgBlurred;                 
	cv::Mat imgThresh;                  
	cv::Mat imgThreshCopy;              

	std::vector<std::vector<cv::Point> > ptContours;        
	std::vector<cv::Vec4i> v4iHierarchy;                    

	cv::Mat Class;     

										
	cv::Mat Flattenedimages;

	
	std::vector<int> ValidChars = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
		'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
		'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
		'U', 'V', 'W', 'X', 'Y', 'Z' };

	TrainingimgNumbers = cv::imread("training_chars.png");

	if (TrainingimgNumbers.empty()) {                               
		std::cout << "image not read from file\n";         
		return(0);                                                 
	}

	cv::cvtColor(TrainingimgNumbers, imgGrayscale, CV_BGR2GRAY);

	cv::GaussianBlur(imgGrayscale,              
		imgBlurred,                             
		cv::Size(5, 5),                         
		0);                                     

												
	cv::adaptiveThreshold(imgBlurred,           
		imgThresh,                              
		255,                                    
		cv::ADAPTIVE_THRESH_GAUSSIAN_C,         
		cv::THRESH_BINARY_INV,                 
		11,                                     
		2);                                     

	cv::imshow("imgThresh", imgThresh);         

	imgThreshCopy = imgThresh.clone();          

	cv::findContours(imgThreshCopy,             
		ptContours,                             
		v4iHierarchy,                           
		cv::RETR_EXTERNAL,                      
		cv::CHAIN_APPROX_SIMPLE);              

	for (int i = 0; i < ptContours.size(); i++) {                           
		if (cv::contourArea(ptContours[i]) > MIN_CONTOUR_AREA) {                
			cv::Rect boundingRect = cv::boundingRect(ptContours[i]);                
			cv::rectangle(TrainingimgN, boundingRect, cv::Scalar(0, 0, 255), 2);

			cv::Mat matROI = imgThresh(boundingRect);           

			cv::Mat matROIResized;
			cv::resize(matROI, matROIResized, cv::Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT));     

			cv::imshow("matROI", matROI);                              
			cv::imshow("matROIResized", matROIResized);                 
			cv::imshow("TrainingImgN", TrainingimgNumbers);

			int intChar = cv::waitKey(0);
			if (intChar == 27) {        
				return(0);             
			}
			else if (std::find(ValidChars.begin(), ValidChars.end(),Char) != ValidChars.end()) {     

				Class.push_back(intChar);       

				cv::Mat matImageFloat;                          
				matROIResized.convertTo(matImageFloat, CV_32FC1);       

				cv::Mat Flattenedimage = matImageFloat.reshape(1, 1);       

				Flattenedimages.push_back(Flattenedimages);       
																							
			}   
		}   
	}   

	std::cout << "training complete\n\n";

	

	cv::FileStorage Class("classifications.xml", cv::FileStorage::WRITE);           

	if (Class.Opened() == false) {                                                        
		std::cout << "unable to open training file\n";        
		return(0);
	}

	Class << "classifications" << Class;        
	Class.release();   


	cv::FileStorage Train("images.xml", cv::FileStorage::WRITE);        

	if (Train.Opened() == false) {                                                
		std::cout << " unable to open training images file\n";         
		return(0);                                                                              
	}

	Train << "images" <<Flattenedimages;       
	Train.release();                                                 

	return(0);
}




