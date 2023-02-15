#ifndef TIMER_HPP
#define TIMER_HPP

#include <chrono>

class Timer {
 public:
  void Start() { start_time_ = std::chrono::steady_clock::now(); }
  int64_t ElapsedMilliSeconds() const {
    auto elapsed = std::chrono::steady_clock::now() - start_time_;
    return std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
  }
  double ElapsedSeconds() const { return ElapsedMilliSeconds() / 1000.0; }

 private:
  std::chrono::steady_clock::time_point start_time_;
};

#endif
