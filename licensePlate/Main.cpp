

#include "Main.h"

int main(void) {

    bool KNNSuccess = TrainKNN();          

    if (KNNSuccess == false) {
        std::cout << std::endl << std::endl << "traning was not successful" << std::endl << std::endl;
        return(0);                                                     
    }

    cv::Mat imgOriginalScene;     

    imgOriginalScene = cv::imread("Detect/100.jpg");        

    if (imgOriginalScene.empty()) {                            
        std::cout << "image not read from file\n\n";     
        _getch();                                               
        return(0);                                            
    }

    std::vector<PossiblePlate> vectorOfPossiblePlates = detectPlatesInScene(imgOriginalScene);         

    vectorOfPossiblePlates = detectCharsInPlates(vectorOfPossiblePlates);                              

    cv::imshow("original image", imgOriginalScene);           

    if (vectorOfPossiblePlates.empty()) {                                               
        std::cout << std::endl << "no license plates were detected" << std::endl;       
    }
    else {                                                                           
                                                                                                                                                             
        std::sort(vectorOfPossiblePlates.begin(), vectorOfPossiblePlates.end(), PossiblePlate::sortDescendingByNumberOfChars);
        PossiblePlate licPlate = vectorOfPossiblePlates.front();

        cv::imshow("imgPlate", licPlate.imgPlate);          
		cv::imshow("imgThresh", licPlate.imgThresh);
        if (licPlate.strChars.length() == 0) {                                                      
            std::cout << std::endl << "no characters were detected" << std::endl << std::endl;     
            return(0);                                                                             
        }

        drawRedRectangleAroundPlate(imgOriginalScene, licPlate);              

        std::cout << std::endl << "Actual license plate read from image = " << licPlate.strChars << std::endl;     
        std::cout << std::endl << "-----------------------------------------" << std::endl;

        writeLicensePlateCharsOnImage(imgOriginalScene, licPlate);              

        cv::imshow("output image", imgOriginalScene);                      

        cv::imwrite("imgOriginalScene.png", imgOriginalScene);                  
    }

    cv::waitKey(0);                

    return(0);
}


void drawRedRectangleAroundPlate(cv::Mat &imgOriginalScene, PossiblePlate &licPlate) {
    cv::Point2f p2fRectPoints[4];

    licPlate.rrLocationOfPlateInScene.points(p2fRectPoints);           

    for (int i = 0; i < 4; i++) {                                      
        cv::line(imgOriginalScene, p2fRectPoints[i], p2fRectPoints[(i + 1) % 4], SCALAR_RED, 2);
    }
}


void writeLicensePlateCharsOnImage(cv::Mat &imgOriginalScene, PossiblePlate &licPlate) {
	cv::Point ptCenterOfTextArea;                 
	cv::Point ptLowerLeftTextOrigin;               

	int intFontFace = CV_FONT_HERSHEY_SIMPLEX;                             
	double dblFontScale = (double)licPlate.imgPlate.rows / 30.0;          
	int intFontThickness = (int)std::round(dblFontScale * 1.5);            
	int intBaseline = 0;

	cv::Size textSize = cv::getTextSize(licPlate.strChars, intFontFace, dblFontScale, intFontThickness, &intBaseline);      

	ptCenterOfTextArea.x = (int)licPlate.rrLocationOfPlateInScene.center.x;         

	if (licPlate.rrLocationOfPlateInScene.center.y < (imgOriginalScene.rows * 0.75)) {      
																							
		ptCenterOfTextArea.y = (int)std::round(licPlate.rrLocationOfPlateInScene.center.y) + (int)std::round((double)licPlate.imgPlate.rows * 1.6);
	}
	else {                                                                               
																						  
		ptCenterOfTextArea.y = (int)std::round(licPlate.rrLocationOfPlateInScene.center.y) - (int)std::round((double)licPlate.imgPlate.rows * 1.6);
	}

	ptLowerLeftTextOrigin.x = (int)(ptCenterOfTextArea.x - (textSize.width / 2));           
	ptLowerLeftTextOrigin.y = (int)(ptCenterOfTextArea.y + (textSize.height / 2));         

																							
	cv::putText(imgOriginalScene, licPlate.strChars, ptLowerLeftTextOrigin, intFontFace, dblFontScale, SCALAR_YELLOW, intFontThickness);
}


