class Sequential {

        VideoCapture* capture;                // OpenCV class to interact with the video
        VideoUtils* videoUtils;               // Class containing the methods to compute frames  
        int width,height;                     // Dimensions of the frames
        unsigned long totFrames;              // Number of frames contained in the video
        unsigned long motionCounter = 0;      // Counter for the frames containing motion 
        int kernelPadding;                    // Derived from kernelSize, used when adding a border to the gray image for the convolution
        Mat* background;                      // Reference to the background frame 

    public:   
        Sequential(const string file_path, int kernelSize, float motionThreshold) { // The constructor sets up the first frame as background
            this->capture = new VideoCapture(file_path);

            this->totFrames = capture->get(CAP_PROP_FRAME_COUNT);
            this->width  = capture->get(CAP_PROP_FRAME_WIDTH);
            this->height = capture->get(CAP_PROP_FRAME_HEIGHT);
            this->kernelPadding = floor(kernelSize / 2);
        
            this->videoUtils = new VideoUtils(width, height, kernelSize, motionThreshold);                              // Will be used in all the frame manipulation
            this->background = new Mat(height, width, CV_8UC1, Scalar(128));                                            // We allocate space for the background frame

        // ------------------------------------------------------------------------------------------------------------------------------------------------------------

            Mat frame;
            Mat* gray = new Mat(height, width, CV_8UC1, Scalar(128));                                                         // We allocate space for the gray version of the background frame
            Mat* paddedGray = new Mat(height + 2*kernelPadding, width + 2*kernelPadding, CV_8UC1, Scalar(128));               // We allocate space for the padded version of the gray background frame
            // We don't allocate space for the smoothed version of the background because Sequential->background will contain it

            capture->read(frame); 
                                                                                                        // Take the fist frame of the video
            videoUtils->chunkwiseGrayscaleConversion(frame, gray, 0, height);                                                 // Tranform the RGB frame into grayscale
            
            copyMakeBorder(*gray, *paddedGray, kernelPadding, kernelPadding, kernelPadding, kernelPadding, BORDER_REPLICATE); // Adds a border to the grayscale background large kernelPadding pixels in all direction
            videoUtils->chunkwiseSmoothing(paddedGray, this->background, 0, height);                                          // Apply smoothing to the background and store it in Sequential->background (redundant)
            
            videoUtils->setBackground(this->background);                                                                      // Copy reference to background in VideoUtils -> will be used in background substraction
            //cout << "Background initialized" << endl;        // Just for debug
            // imwrite("results/bs.png", *background);          // Just for debug -> 'b' for background, 's' for sequential
            
            // Release memory previously allocated
            delete gray;              
            delete paddedGray;
        }

        ~Sequential() { // Destructor -> Release memory previously allocated while shutting down the program
            capture->release();
            delete background;
            delete capture;
            delete videoUtils;
        }

        void run() { // Main method -> Perform (sequentially) the whole workflow
            Mat frame;                                                                                                                  // Will contain the frames extracted from the video
            Mat* gray = new Mat(height, width, CV_8UC1, Scalar(128));                                                                   // We allocate space for the gray version of each frame
            Mat* smooth = new Mat(height, width, CV_8UC1, Scalar(128));                                                                 // We allocate space for the smoothed version of each frame
            Mat* paddedGray = new Mat(height + 2*kernelPadding, width + 2*kernelPadding, CV_8UC1, Scalar(128));                         // We allocate space for the padded version of each frame
            for (int i = 1; i < totFrames; i ++){ // Starts by 1 to account for the background
                
                capture->read(frame);                                                                                                   // Read each frame
                videoUtils->chunkwiseGrayscaleConversion(frame, gray, 0, height);                                                       // Tranform the RGB frame into grayscale
                                
                copyMakeBorder(*gray, *paddedGray, kernelPadding, kernelPadding, kernelPadding, kernelPadding, BORDER_REPLICATE);       // Add borders
                videoUtils->chunkwiseSmoothing(paddedGray, smooth, 0, height);                                                          // Perform smoothing

                motionCounter += videoUtils->detectMotion(smooth);                                                                      // (Potentially) Increase the counter for frames containing motion

                // imwrite("results/" + to_string(i) + "s.png", *smooth);       // just for debug
                
            }
            
            cout << "Frames containing motion : " << motionCounter << " / " << totFrames << endl;
            
            // Release memory
            delete gray;
            delete paddedGray;
            delete smooth;
            exit(0);
        }

        void runTimer() { // Mostly the same workflow as in run() -> refer to the comments above
            Mat frame;
            Mat* gray = new Mat(height, width, CV_8UC1, Scalar(128));
            Mat* smooth = new Mat(height, width, CV_8UC1, Scalar(128));
            Mat* paddeGray = new Mat(height + 2*kernelPadding, width + 2*kernelPadding, CV_8UC1, Scalar(128));
            
            // Create an isolated scope to exploit utimer destructor properties
            {
                utimer timer("Sequential implementation", false);                    // for debug
                           
                for (int i = 1; i < totFrames; i ++){
                
                    capture->read(frame);
                    videoUtils->chunkwiseGrayscaleConversion(frame, gray, 0, height);  
                                    
                    copyMakeBorder(*gray, *paddeGray, kernelPadding, kernelPadding, kernelPadding, kernelPadding, BORDER_REPLICATE);       
                    videoUtils->chunkwiseSmoothing(paddeGray, smooth, 0, height);

                    motionCounter += videoUtils->detectMotion(smooth);

                }
            }       

            delete gray;
            delete paddeGray;
            delete smooth;
            exit(0);
        }

        void runDetailedTimer() { // Mostly the same workflow as in run() -> refer to the comments above
            Mat frame;
            Mat* gray = new Mat(height, width, CV_8UC1, Scalar(128));
            Mat* smooth = new Mat(height, width, CV_8UC1, Scalar(128));
            Mat* paddeGray = new Mat(height + 2*kernelPadding, width + 2*kernelPadding, CV_8UC1, Scalar(128));

            unsigned long grayTime = 0, smoothTime = 0, detectTime = 0;
            long elapsed;
            
            // Create an isolated scope to exploit utimer destructor properties
            {
                // utimer timer("Sequential implementation"); 
                utimer timer("Sequential implementation", false);                    // for debug
                           
                for (int i = 1; i < totFrames; i ++){
                
                    capture->read(frame);

                    
                    {
                        utimer grayTimer("grayscale conversion", &elapsed, false);
                        videoUtils->chunkwiseGrayscaleConversion(frame, gray, 0, height);  
                    }
                    grayTime += elapsed;
                    
                    {
                        utimer smoothTimer("smoothing process", &elapsed, false);
                        copyMakeBorder(*gray, *paddeGray, kernelPadding, kernelPadding, kernelPadding, kernelPadding, BORDER_REPLICATE);       
                        videoUtils->chunkwiseSmoothing(paddeGray, smooth, 0, height);
                    }
                    smoothTime += elapsed;
                    
                    {
                        utimer detectTimer("background comparison and motion detection", &elapsed, false);
                        motionCounter += videoUtils->detectMotion(smooth);
                    }
                    detectTime += elapsed;
                    
                }
            }

            // cout << "Gray-scale conversion computed in " << grayTime << " microseconds " << endl;
            cout << grayTime << endl;
            // cout << "Smoothing process computed in " << smoothTime << " microseconds " << endl;
            cout << smoothTime << endl;
            // cout << "Background comparison and Motion detection computed in " << detectTime << " microseconds " << endl;
            cout << detectTime << endl;

            delete gray;
            delete paddeGray;
            delete smooth;
            exit(0);
        }
};