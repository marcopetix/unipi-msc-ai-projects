## ⚡ Parallel and Distributed Systems: Paradigms and Models (University of Pisa, 21/07/2022)

This folder contains my final project for the *Parallel and Distributed Systems: Paradigms and Models* course at the University of Pisa.  
The project implements and analyzes a motion detection system for videos, developed in three architectures: **sequential**, **thread-based**, and **FastFlow-based parallel** versions.

---

### 📄 Final Project: Video Motion Detection System

- **Goal:**  
  Build a motion detection pipeline for video streams, comparing sequential and parallel implementations in terms of speedup, efficiency, and scalability.

- **Pipeline steps:**  
  ✅ Frame extraction (OpenCV)  
  ✅ Grayscale conversion  
  ✅ Smoothing via convolutional kernel  
  ✅ Background comparison  
  ✅ Motion detection based on pixel difference thresholds

---

### 🛠️ Architectures implemented

- **Sequential version:**  
  Baseline version processing frames one by one on a single core.

- **Thread-based parallel version:**  
  Custom C++ thread pool, with:
  - Inter-frame parallelization (multiple frames processed in parallel)  
  - Optional intra-frame parallelization (splitting per-frame work into async chunks)  
  - Optional thread pinning to CPU cores

- **FastFlow-based version:**  
  Stream-parallel Farm pattern with Emitter → Workers → Collector, leveraging FastFlow for efficient multicore processing.

---

### 🔬 Key experiments and analyses

- Bottleneck analysis on grayscale, smoothing, and detection phases  
- Thread pinning impact evaluation  
- Intra-frame parallelization tests (using async)  
- Scalability benchmarks on varying kernel sizes and worker counts  
- Speedup and efficiency comparisons between thread-based and FastFlow implementations

---

### 🏆 Outcome

Final grade: **30/30**

---

### 💡 Key learning points

- Hands-on experience with multicore programming paradigms  
- Stream-parallel design with FastFlow  
- Performance benchmarking and optimization  
- Experimental analysis of parallel scalability and resource bottlenecks

---

### 📂 Structure
/SPM \
├── demo/ \
│ ├── media/ \
│ ├── FastFlow.cpp \
│ ├── launcher.cpp \
│ ├── Parallel.cpp \
│ ├── Sequential.cpp \
│ ├── utimer.cpp \
│ └── VideoUtils.cpp \
├── ff/ \
├── README.md \
└── SPM_Report.pdf \
