// Including libraries
#include "opencv2/opencv.hpp" // OpenCV for the extraction of frames from the videos
#include <ff/ff.hpp> // FastFlow library
#include <chrono> // Used in utimer.cpp 
#include <iostream>
#include <vector>
#include <queue>
#include <atomic> // For the atomic counter -> MotionCounter
#include <thread>
#include <future>
#include <mutex>

// Namespace to lighten the amount of code
using namespace std;	
using namespace cv;
using namespace ff;

// Include other files
#include "utimer.cpp"     // Just as seen during the lectures -> contains methods to measure computation times
#include "VideoUtils.cpp" // Contains all the frame manipulation methods
#include "Sequential.cpp" // Contains the sequential implementation of the project
#include "Parallel.cpp"   // Contains the thread-based implementation of the project
#include "FastFlow.cpp"   // Contains the FastFlow implementation of the project

int main(int argc,char* argv[]) {

    if (argc < 8) {
        cout << "Arguments :" << endl;
        cout << "1 - File path to the video : es. 'demo/sloth_a.mp4'" << endl;
        cout << "2 - Version of the program: (0) Sequential, (1) Threads, (2) FastFlow" << endl;
        cout << "3 - ShowTimers : (0) No, (1) Yes, (2) Detailed timer (Sequential-only)" << endl;
        cout << "4 - Percentage threshold for the motion detection: es. [0.01 - 1.0]" << endl;
        cout << "5 - Number of (inter-frame) paralell workers: es. [1, 2, ...] " << endl;
        cout << "6 - Number of (intra-frame) paralell workers: es. [1, 2, ...] " << endl;
        cout << "7 - Thread Pinning : (0) No, (1) Yes" << endl;
        cout << "8 - KernelSize for the smoothing process: 3 by default (must be odd)" << endl;
        
        exit(-1);
    }

	string filePath       = argv[1];                   // File path of the video, Ex: media/demo_a.mp4
    int version            = atoi(argv[2]);             // Version of the program : (0) Sequential, (1) Threads and (2) FastFlow 
    int showTimers         = atoi(argv[3]);             // Print timers obtained thanks to utimer.cpp : (0) No timers, (1) with timers
    float motionThreshold  = atof(argv[4]);             // Percentage of differing pixels triggering the motion detection, Ex: 0.3 
    int interFrameWorkers  = atoi(argv[5]);             // Number of thread workers for the thread pool
    int intraFrameWorkers  = atoi(argv[6]);             // Number of threads/async Split the computation of the grayscale conversion and smoothing application for each frame 
    int threadPinning      = atoi(argv[7]);             // Enables thread pinning : (0) No , (1) yes
    int kernelSize = (argc == 8) ? 3 : atoi(argv[8]);   // Dimension of the kernel used in the smoothing convolution, Ex: kernelSize = 3 leads to a 3x3 convolution 

    if ( ( version < 0 ) || ( version > 2 ) ) { cout << "Insert valid value for 'version' : [0,1,2]" << endl; exit(-1); }
    if ( ( showTimers < 0 ) || ( showTimers > 2 ) ) { cout << "Insert valid value for 'showTimers' : [0, 1]" << endl; exit(-1); }
    if ( ( motionThreshold < 0 ) || ( motionThreshold > 1 ) ) { cout << "Insert a float value for 'motionThreshold' : [0.0-1.0]" << endl; exit(-1); }
    if ( interFrameWorkers < 0) { cout << "Insert a positive value for 'interFrameWorkers' [1,...]" << endl; exit(-1); }
    if ( intraFrameWorkers < 0) { cout << "Insert a positive value for 'intraFrameWorkers' [1,...]" << endl; exit(-1); }
    if ( ( threadPinning < 0 ) || ( threadPinning > 1 ) ) { cout << "Insert valid value for 'threadPinning' : [0, 1]" << endl; exit(-1); }
    if ( ( kernelSize % 2 == 0) || ( kernelSize < 0 ) ) { cout << "Insert positive and odd value for 'kernelSize' : [3, 5, 7,...]" << endl; exit(-1); }

    if (version == 0) { 
        //cout << "Sequential version in 3, 2, 1..." << endl;
        Sequential seq(filePath, kernelSize, motionThreshold); // The constructor takes care to extract and prepare the background 
        if (showTimers == 0) seq.run();
            else if (showTimers == 1) seq.runTimer();
                else seq.runDetailedTimer();
	}

    if (version == 1) { 
        // cout << "Parallel version (" << interFrameWorkers << ", " << intraFrameWorkers << ") in 3, 2, 1..." << endl;
        Parallel par(filePath, kernelSize, motionThreshold, interFrameWorkers, intraFrameWorkers, threadPinning);  
        if (showTimers) par.runTimer();
            else par.run();
    }

    if (version == 2) { 
        // cout << "FastFlow version (" << interFrameWorkers << ") in 3, 2, 1..." << endl;
        FastFlow ff(filePath, kernelSize, motionThreshold, interFrameWorkers); 
        if (showTimers) ff.runTimer();
            else ff.run();
    }

    return 0;
}