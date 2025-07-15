atomic<ulong> motionCounter(0); // Atomic counter used to keep track of the frames containing motion

class Parallel { // The constructor sets up the first frame as background -> Mostly similar to the Sequential constructor  

    VideoCapture* capture;                                  // OpenCV class to interact with the video
    VideoUtils* videoUtils;                                 // Class containing the methods to compute frames  
    int height, width;                                      // Dimensions of the frames
    unsigned long totFrames;                                // Number of frames contained in the video
    int kernelPadding;                                      // Derived from kernelSize, used when adding a border to the gray image for the convolution
    Mat* background;                                        // Reference to the background frame 

    // Variables for the implementation of thread pool
    unsigned int interFrameWorkers;                         // Dimension of the thread pool used in the implementation of the farm 
    unsigned int intraFrameWorkers;                         // Number of async utilized to split the intra-frame work with the thread -> See the report about it
                                                                // Just implemented as a mean to practice and experiment 
                                                                // Not tested on the remote machine due to time constraints
    mutex queueMutex;                                       // Used when submitting/extracting tasks to/from the queue
    mutex iomutex;                                          // Used (for debug reasons) when printing on the iostream  
    deque<packaged_task<void()>> tasks;                     // Double-ended queue containing the tasks (tasks : worker function + frame)
    condition_variable cond;                                // Used to notify the threads of changes in the queue
    bool stop;                                              // Used for the stopping condition of the thread pool
    
    vector <thread> threadIds;                              // Vector of threads making up the thread pool
    vector <future<void>> threadFutures;                    // Vetore of futures used to assure the computation of all the frames
    bool threadPinning;                                     // Determine if the threads from the threadpool are to be pinned to single cores
    
    public:
        Parallel(const string file_path, int kernelSize, float motionThreshold, int interFrameWorkers, int intraFrameWorkers, int ThreadPinning) {
            this->capture = new VideoCapture(file_path);
            
            this->totFrames = capture->get(CAP_PROP_FRAME_COUNT);
            this->width = capture->get(CAP_PROP_FRAME_WIDTH);
            this->height = capture->get(CAP_PROP_FRAME_HEIGHT);

            this->kernelPadding = floor(kernelSize / 2);
            this->interFrameWorkers = interFrameWorkers;
            this->intraFrameWorkers = intraFrameWorkers;

            this->videoUtils = new VideoUtils(width, height, kernelSize, motionThreshold);                          // Will be used in all the frame manipulation
            this->background = new Mat(height, width, CV_8UC1, Scalar(128));                                        // We allocate space for the background frame

            this->stop = false;
            this->threadPinning = threadPinning;

            // ------------------------------------------------------------------------------------------------------------------------------------------------------------

            Mat frame;
            Mat* gray = new Mat(height, width, CV_8UC1, Scalar(128));                                               // We allocate space for the gray version of the background frame
            Mat* paddedGray = new Mat(height + 2*kernelPadding, width + 2*kernelPadding, CV_8UC1, Scalar(128));     // We allocate space for the padded version of the gray background frame
            // We don't allocate space for the smoothed version of the background because Parallel->background will contain it

            capture->read(frame);                                                                                   // Read the fist frame of the video
            videoUtils->chunkwiseGrayscaleConversion(frame, gray, 0, height);                                       // Tranform the RGB frame into grayscale
            
            copyMakeBorder(*gray, *paddedGray, kernelPadding, kernelPadding, kernelPadding, kernelPadding, BORDER_REPLICATE);   // Adds a border to the grayscale background large kernelPadding pixels in all direction
            videoUtils->chunkwiseSmoothing(paddedGray, this->background, 0, height);                                // Apply smoothing to the background and store it in Parallel->background (redundant)
            
            videoUtils->setBackground(this->background);                                                            // Copy reference to background in VideoUtils -> will be used in background substraction
            // cout << "Background initialized" << endl;        // Just for debug 
            // imwrite("results/bp.png", *background);          // Just for debug -> 'b' for background, 'p' for sequential
            
            // Release memory previously allocated
            delete gray; 
            delete paddedGray;
        }

        ~Parallel() { // Destructor -> Release memory previously allocated while shutting down the program
            capture->release();
            delete background;
            delete capture;
            delete videoUtils; 
        }

        void threadWorker(Mat frame) { // Splits the work for the grayscale conversion and the smoothing between the thread and some async helpers 
                                       // If intraFrameWorkers = 1 then it act as the default sequential worker that is described in the report
                                       // The experiments performed on the remote machine fall under this case (interFrameWorkers = n, intraFrameWorkers = 1)
            
            Mat* gray = new Mat(height, width, CV_8UC1, Scalar(128));
            Mat* paddedGray = new Mat(height + 2*kernelPadding, width + 2*kernelPadding, CV_8UC1, Scalar(128));
            Mat* smooth = new Mat(height, width, CV_8UC1, Scalar(128));
            
            vector <pair<int,int>> splittingPoints(intraFrameWorkers); // Will contain the start and end points of each chunk of rows
            vector <future<void>> helpersFuture(intraFrameWorkers-1);  // Will contain the futures of the async helpers so to prompt them to perform the computation in time 

            for(int i=0; i<intraFrameWorkers; i++) {
                splittingPoints[i].first = (height * i) / intraFrameWorkers;
                splittingPoints[i].second = ((height * (i + 1)) / intraFrameWorkers) - 1;
            }

            for(int i=0; i<intraFrameWorkers-1; i++) { // We instantiate the async helpers and assing them a chunk
                helpersFuture[i] = async(launch::async, [&]() { videoUtils->chunkwiseGrayscaleConversion(frame, gray, splittingPoints[i].first, splittingPoints[i].second); } );
            }
            
            // The thread worker compute the last chunk or the entire image if intraFrameWorkers = 1
            videoUtils->chunkwiseGrayscaleConversion(frame, gray, splittingPoints[intraFrameWorkers-1].first, splittingPoints[intraFrameWorkers-1].second);  
            for(int i=0; i<intraFrameWorkers-1; i++) {
                helpersFuture[i].get();           // The future of the async is used to make sure that the computation is performed before reading from the matrix
            }
                                
            copyMakeBorder(*gray, *paddedGray, kernelPadding, kernelPadding, kernelPadding, kernelPadding, BORDER_REPLICATE);       
            
            
            for(int i=0; i<intraFrameWorkers-1; i++) {
                helpersFuture[i] = async(launch::async, [&]() { videoUtils->chunkwiseSmoothing(paddedGray, smooth, splittingPoints[i].first, splittingPoints[i].second); } );
            }
            
            videoUtils->chunkwiseSmoothing(paddedGray, smooth, splittingPoints[intraFrameWorkers-1].first, splittingPoints[intraFrameWorkers-1].second);  
            for(int i=0; i<intraFrameWorkers-1; i++) {
                helpersFuture[i].get();           // The future of the async is used to make sure that the computation is performed before reading from the matrix
            }

            motionCounter += videoUtils->detectMotion(smooth);  // The thread worker performs the increment on the atomic counter

            delete gray;
            delete paddedGray;
            delete smooth;
        };

        // Mostly similar to what has been seen during the lectures -> Main workflow of the threads
        void body() { 
            while(true) {
                packaged_task <void()> task;
                {
                    unique_lock<mutex> lock(queueMutex);                            // unique lock for RAII 
                    cond.wait(lock, [this]() { return(!tasks.empty() || stop); });  // Stops when the queue is empty or when the job is done

                    if (!tasks.empty()) {
                        task = move(tasks.front());                                 // move() is needed to interact with the packaged_task
                        tasks.pop_front();                                          // tasks.front() doesn't remove anything, we need to remove with pop_front()
                    }

                    if(stop)
                        return;
                }
                task();
            }
        };

        void submit(packaged_task<void()> & task) {         // the task to be submitted is a bind object made up of a frame and a function for the thread 
            {
                unique_lock<mutex> lock(queueMutex);        // Unique_lock for RAII
                tasks.push_back(move(task));                // Adds new task to the queue
            }
            cond.notify_one();                              // Notifies one of the threads waiting
        };

        void stopThreadPoool() {                            // Stops the thread pool by triggering the stopping condition
            {
                unique_lock<mutex> lock(queueMutex);        // Unique_lock for RAII
                stop = true;
            }
            cond.notify_all();                              // We notifies all the threads to check the stopping condition
        };

        void run() {

            if (!threadPinning) {
                // Threads are added to the pool without regard for their core affinity
                for(int i=0; i<interFrameWorkers; i++) 
                    threadIds.push_back(thread( [this] () { this->body(); } ));                 // The [this] in the lambda is necessary to access elements of the Paralllel class
            }else{
                // Threads are added to the pool by pinning each of them to a single core
                unsigned short numCPU = thread::hardware_concurrency();                         // Gets the number of cores
                for(int i=0; i<interFrameWorkers; i++){
                    threadIds.push_back(thread( [this] () { this->body(); } )); 
                    cpu_set_t cpuset;
                    CPU_ZERO(&cpuset);
                    CPU_SET(i % numCPU, &cpuset);                                               // Threads are pinned to cores in a "round-robin-like" fashion 
                    int rc = pthread_setaffinity_np(threadIds[i].native_handle(), sizeof(cpu_set_t), &cpuset); // Create a mask that pins the thread to a single core
                    if (rc != 0) std::cerr << "Error calling pthread_setaffinity_np: " << rc << "\n";
                }
            }

            // We prepare the packaged_tasks
            for(int i=1; i<totFrames; i++) {
                
                Mat frame; 
                capture->read(frame);                                                                   // We read the frames

                
                auto fx = (bind([this] (Mat frame) { this->threadWorker(frame); }, frame));   

                packaged_task<void()> pt(fx);                                                           // We put the bind object in the packaged_task
                threadFutures.push_back(pt.get_future());                                               // We store the futures of each task
                submit(pt);                                                                             // We submit the task
            }

            // The futures are called for each task (for each frame) to prompt the threads to compute the output
            for(int i=0; i<totFrames-1; i++) { // We stop at totFrames -1 to account for the background
                threadFutures[i].get();
            }

            // After computing the output the thread pool is stopped
            stopThreadPoool();

            // We finally join each of the threads
            for(int i=0; i<interFrameWorkers; i++)
                threadIds[i].join();

            cout << "Frames containing motion : " << motionCounter << " / " << totFrames << endl;

            exit(0);
        }

        void runTimer() {  // Mostly the same workflow as in run() -> refer to the comments above
 
            Mat frame;

            // Create an isolated scope to exploit utimer destructor properties
            {
                // utimer timer("Parallel implementation");
                utimer timer("Parallel implementation", false);                    // for debug

                if (threadPinning) {
                    unsigned short numCPU = thread::hardware_concurrency();
                    for(int i=0; i<interFrameWorkers; i++){
                        threadIds.push_back(thread( [this] () { this->body(); } )); 
                        cpu_set_t cpuset;
                        CPU_ZERO(&cpuset);
                        CPU_SET(i % numCPU, &cpuset);
                        int rc = pthread_setaffinity_np(threadIds[i].native_handle(), sizeof(cpu_set_t), &cpuset);
                        if (rc != 0) {
                            std::cerr << "Error calling pthread_setaffinity_np: " << rc << "\n";
                        } 
                    }
                }else{
                    for(int i=0; i<interFrameWorkers; i++) 
                        threadIds.push_back(thread( [this] () { this->body(); } )); 
                }

                for(int i=0; i<totFrames-1; i++) {
                    capture->read(frame);
                    auto fx = (bind([this] (Mat frame) { this->threadWorker(frame); }, frame));
                    packaged_task<void()> pt(fx);
                    threadFutures.push_back(pt.get_future()); 
                    submit(pt);
                }

                for(int i=0; i<totFrames-1; i++) {
                    threadFutures[i].get();
                }

                stopThreadPoool();

                for(int i=0; i<interFrameWorkers; i++)
                    threadIds[i].join();

            }

            exit(0);
        }
  
};