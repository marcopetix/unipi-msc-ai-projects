class VideoUtils {
    public:
        int width,height;           // Dimensions of the frames
        int totPixels;              // Total number of pixels in the frames -> used when computing percentage of different pixels
        int kernelSize;             // Lenght of the side of the convolution kernel
        int kernelArea;             // Size of the convolution kernel
        int kernelPadding;          // Derived from kernelSize, used as offset during convolution
        float motionThreshold;      // Percentage of differing pixels triggering the motion detection
        Mat* background;            //Address of the background frame (used in background comparison) 

        VideoUtils(int width, int height, int kernelSize, float motionThreshold): 
            width(width), height(height), totPixels(width*height),
            kernelSize(kernelSize), kernelArea(kernelSize * kernelSize), kernelPadding(floor(kernelSize / 2)),
            motionThreshold(motionThreshold), background(nullptr) { }

        void setBackground(Mat* background) {
            this->background = background;
        }

        // Deprecated -> Passed to chunkwiseGrayscaleConversion()
        void grayscaleConversion(const Mat colorImage, Mat* grayImage) { 
            float red, green, blue;
            int i, j, gray;

            for (i = 0; i < height; i++) {
                for (j = 0; j < width; j++){
                    
                    //OpenCV sorts colors for RGB as BGR
                    blue = colorImage.at<Vec3b>(i, j).val[0]; 
                    green = colorImage.at<Vec3b>(i, j).val[1];
                    red = colorImage.at<Vec3b>(i, j).val[2];
                    
                    gray = round( (blue + green + red) / 3 );
                    grayImage->at<uchar>(i, j) = gray;
                }
            }
        }

        // chunkwise version of the method as been implemented to experiment with intra-frame parallelization
        void chunkwiseGrayscaleConversion(const Mat colorImage, Mat* grayImage, int start, int end) {
            float red, green, blue;
            int i, j, gray;

            // Works row-by-row -> It's good for the cache because OpenCV stores matrices by rows 
            for (i = start; i < end; i++) {
                for (j = 0; j < width; j++){

                    //OpenCV sorts colors for RGB as BGR -> Don't really know why
                    blue = colorImage.at<Vec3b>(i, j)[0]; 
                    green = colorImage.at<Vec3b>(i, j)[1];
                    red = colorImage.at<Vec3b>(i, j)[2];
                    
                    gray = round( (blue + green + red) / 3 );
                    grayImage->at<uchar>(i, j) = gray;
                }
            }
            
        }

        // Deprecated -> Passed to chunkwiseSmoothing() 
        void Smoothing(Mat* grayImage, Mat* smoothedImage) { 
            int i, j, sum;

            for (i = kernelPadding; i < height + kernelPadding; i++) {
                for (j = kernelPadding; j < width + kernelPadding; j++){

                    sum = 0;
                    
                    for (int x = i - kernelPadding; x <= i + kernelPadding; x++){
                        for (int y = j - kernelPadding; y <= j + kernelPadding; y++){
                            sum += grayImage->at<uchar>(x, y);
                        }
                    }
                    smoothedImage->at<uchar>(i - kernelPadding, j - kernelPadding) = sum / kernelArea;
                }
            }
            
        }

        void chunkwiseSmoothing(Mat* grayImage, Mat* smoothedImage, int start, int end) {
            int i, j, sum;

            // Works row-by-row but skips the padding borders 
            for (i = start + kernelPadding; i < end + kernelPadding; i++) {
                for (j = kernelPadding; j < width + kernelPadding; j++){

                    sum = 0;

                    // Internal cycle performs the convolution        
                    for (int x = i - kernelPadding; x <= i + kernelPadding; x++){
                        for (int y = j - kernelPadding; y <= j + kernelPadding; y++){
                            sum += grayImage->at<uchar>(x, y);
                        }
                    }
                    // KernelPadding also acts as offset for wrt to the standard-sized smooth images
                    smoothedImage->at<uchar>(i - kernelPadding, j - kernelPadding) = sum / kernelArea;
                }
            }
            
        }

        unsigned short detectMotion(Mat * smoothedImage) {
            
            unsigned long differentPixels = 0; 

            // Pixel-wise background comparison
            // Works row-by-row -> It's good for the cache because OpenCV stores matrices by rows 
            for (int i = 0; i < height; i++) { 
                for (int j = 0; j < width; j++){
                    differentPixels += ( background->at<uchar>(i, j) != smoothedImage->at<uchar>(i,j) );
                }
            }

            float differentPixelsPercentage = float(differentPixels) / totPixels;
            return differentPixelsPercentage >= motionThreshold; // 0 or 1 are directly used as potential increments for the motionCounter
        }

};