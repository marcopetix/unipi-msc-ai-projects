// The same class showed during the lectures
// Changes : 
//    - Removed the std:: reference due to the namespace inclusion in launcher.cpp 
//    - Added bool verbose property for debug reasons
class utimer {
  chrono::system_clock::time_point start;
  chrono::system_clock::time_point stop;
  string message;
  bool verbose;
  using usecs = chrono::microseconds;
  using msecs = chrono::milliseconds;

private:
  long * us_elapsed;
  
public:

  utimer(const string m, bool verbose) : message(m), verbose(verbose), us_elapsed((long *)NULL) {
    start = chrono::system_clock::now();
  }
    
  utimer(const string m, long * us, bool verbose) : message(m), verbose(verbose), us_elapsed(us) {
    start = chrono::system_clock::now();
  }

  ~utimer() {
    stop = chrono::system_clock::now();
    chrono::duration<double> elapsed = stop - start;
    
    auto usec = chrono::duration_cast<usecs>(elapsed).count();

    if (verbose) cout << message << " computed in " << usec << " microseconds " << endl;
       else cout << usec << endl;
    if(us_elapsed != NULL){
      (*us_elapsed) = usec;
    } 
  }
};