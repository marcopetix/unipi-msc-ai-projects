class FarmSource : public ff_monode_t<void*,Mat> { //Source / Emitter of the stream of frames
    public:        
        VideoCapture* capture;                // OpenCV class to interact with the video
        unsigned long totFrames;              // Number of frames contained in the video
        int height, width, nbytes;            // Dimensions of the frames

        FarmSource(VideoCapture* capture) : capture(capture) {}

        int svc_init() {                      // Initialize the class properties
            this->totFrames = capture->get(CAP_PROP_FRAME_COUNT);
            this->width  = capture->get(CAP_PROP_FRAME_WIDTH);
            this->height = capture->get(CAP_PROP_FRAME_HEIGHT);
            this->nbytes = sizeof(unsigned char)*width*height*3;
            return(0);
        }
        
        Mat* svc(void **) { 
            for(int i=1; i<totFrames; i++) {                                        // We start from i=1 to account for the background
                Mat frame;
                capture->read(frame);                                               // We read the frame
                Mat* framePointer = new Mat(height,width,CV_8UC3,Scalar(0,0,0));    // Allocate memory for the frame
                memcpy(framePointer->data, frame.data, nbytes);                     // Copy the frame in said memory 
                // imwrite("results/"+ to_string(i)+"ff.png", *framePointer);       // Just for debug

                ff_send_out(framePointer);                                          // Send the frames to the Workers nodes
            }
            return EOS;                                                             // Send status update at the end of the stream
        } 

        void svc_end(){
            // cout << "FarmSource ended" << endl;                                  // Just for debug
        }
};

class FarmWorker : public ff_node_t<Mat, unsigned short> { // The worker nodes doing most of the computation

    public:  
        VideoUtils* videoUtils;                            // Class containing the methods to compute frames  
        int height, width, kernelPadding;                  // Dimensions of the frames and border to apply for convolution
        Mat* gray;                                         // We will store here the images after the grayscale conversion
        Mat* smooth;                                       // We will store here the images after the convolution
        Mat* paddedGray;                                   // We will store here the images after adding the borders

        FarmWorker(VideoUtils* videoUtils) {
            this->videoUtils = videoUtils;
            this->height = videoUtils->height;
            this->width = videoUtils->width;
            this->kernelPadding = videoUtils->kernelPadding;
            gray = new Mat(height, width, CV_8UC1, Scalar(128));                                            // Allocate memory for the image
            smooth = new Mat(height, width, CV_8UC1, Scalar(128));                                          // Allocate memory for the image
            paddedGray = new Mat(height + 2*kernelPadding, width + 2*kernelPadding, CV_8UC1, Scalar(128));  // Allocate memory for the image
         }

        unsigned short* svc(Mat* frame) {
            videoUtils->chunkwiseGrayscaleConversion(*frame, gray, 0, height);              // Apply grayscale conversion
            copyMakeBorder(*gray, *paddedGray, kernelPadding, kernelPadding, kernelPadding, kernelPadding, BORDER_REPLICATE);  // Add borders to gray image
            videoUtils->chunkwiseSmoothing(paddedGray, smooth, 0, height);                  // Apply convolution

            return new unsigned short( videoUtils->detectMotion(smooth) );                  // (Potentially) Increases the motionCounter
            
        }

        void svc_end() {
            // cout << "FarmWorker ended" << endl;          // Just for debug
            // Releases the allocated memory
            delete gray;
            delete paddedGray;
            delete smooth;
        }

};

class FarmSink : public ff_minode_t<unsigned short> { // Sink / Collector of the farm
    public:
        unsigned long * motionCounter;                      // Reference to the counter of frames containing motion

        FarmSink(unsigned long * motionCounter) : motionCounter(motionCounter) {}

        unsigned short* svc(unsigned short* motionFlag) {   // Increases the counter with the results from the workers computation
            *motionCounter += *motionFlag;
            delete motionFlag;                              // Free the allocated memory for the workers results
            return GO_ON;
        } 

        void svc_end() {
            // cout << "FarmSink ended" << endl;            // Just for debug
        }
};

class FastFlow {

        VideoCapture* capture;             // OpenCV class to interact with the video
        VideoUtils* videoUtils;            // Class containing the methods to compute frames  
        int width,height;                  // Dimensions of the frames
        unsigned long totFrames;           // Number of frames contained in the video
        int kernelPadding;                 // Derived from kernelSize, used when adding a border to the gray image for the convolution
        unsigned long motionCounter = 0;   // Counter for the frames containing motion 
        int nWorkers;                      // Number of the worker nodes in the implementation of the farm 
        Mat* background;                   // Background frame used in the background comparison        

    public:   
        FastFlow(const string file_path, int kernelSize, float motionThreshold, int nWorkers) { // constructor sets up the first frame as background
            this->capture = new VideoCapture(file_path);
            
            this->totFrames = capture->get(CAP_PROP_FRAME_COUNT);
            this->width = capture->get(CAP_PROP_FRAME_WIDTH);
            this->height = capture->get(CAP_PROP_FRAME_HEIGHT);
            
            this->kernelPadding = floor(kernelSize / 2);
            this->nWorkers = nWorkers;

            this->videoUtils = new VideoUtils(width, height, kernelSize, motionThreshold);
            this->background = new Mat(height, width, CV_8UC1, Scalar(128));

        // ------------------------------------------------------------------------------------------------------------------//

            Mat frame;
            Mat* gray = new Mat(height, width, CV_8UC1, Scalar(128));
            Mat* paddedGray = new Mat(height + 2*kernelPadding, width + 2*kernelPadding, CV_8UC1, Scalar(128));

            capture->read(frame);                                              
            videoUtils->chunkwiseGrayscaleConversion(frame, gray, 0, height);  
            
            copyMakeBorder(*gray, *paddedGray, kernelPadding, kernelPadding, kernelPadding, kernelPadding, BORDER_REPLICATE);
            videoUtils->chunkwiseSmoothing(paddedGray, this->background, 0, height); 
            
            videoUtils->setBackground(this->background);                                
            // cout << "Background initialized" << endl;    // Just for debug
            // imwrite("results/bff.png", *background);     // Just for debug
            
            delete gray; 
            delete paddedGray;
        }

        ~FastFlow() {   // Destructor -> Release memory previously allocated while shutting down the program
            capture->release();
            delete background;
            delete capture;
            delete videoUtils;
        }

        void run() {
            //declare farm
            ff_farm farm;
            
            //declare source node
            FarmSource source(capture);
            farm.add_emitter(&source);
            
            //declare vector of workers node
            vector <ff_node*> farmWorkers;
            for(int i=0; i<nWorkers; i++)  farmWorkers.push_back(new FarmWorker(videoUtils));
            
            //add workers to the farm
            farm.add_workers(farmWorkers);

            //declare sink node
            FarmSink sink(&motionCounter);
            farm.add_collector(&sink);
            

            //set scheduling as "on demand" -> reduce the worker's input buffer size to better balance the work load
            farm.set_scheduling_ondemand();

            //run the farm
            farm.run_and_wait_end();

            cout << "Frames containing motion : " << motionCounter << " / " << totFrames << endl;

            exit(0);
        }

        void runTimer() {
            //declare farm
            ff_farm farm;
            
            //declare source node
            FarmSource source(capture);
            farm.add_emitter(&source);
            
            //declare vector of workers node
            vector <ff_node*> farmWorkers;
            for(int i=0; i<nWorkers; i++)  farmWorkers.push_back(new FarmWorker(videoUtils));
            
            //add workers to the farm
            farm.add_workers(farmWorkers);

            //declare sink node
            FarmSink sink(&motionCounter);
            farm.add_collector(&sink);
            

            //set scheduling as "on demand" -> reduce the worker's input buffer size to better balance the work load
            farm.set_scheduling_ondemand();

            // Create an isolated scope to exploit the destructor of the utimer class
            {
                utimer timer("FastFlow implementation", false);
                //run the farm
                farm.run_and_wait_end();
            }
            

            //clean what needs to be cleaned
            exit(0);
        }

};