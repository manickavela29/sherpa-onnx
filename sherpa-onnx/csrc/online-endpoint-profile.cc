// sherpa-onnx/csrc/online-endpoint-profile.cc
//
// Copyright (c)  2026
#include "sherpa-onnx/csrc/online-endpoint-profile.h"

#include <chrono>  // NOLINT
#include <condition_variable>  // NOLINT
#include <cstdio>
#include <cstdlib>
#include <mutex>  // NOLINT
#include <thread>  // NOLINT
#include <vector>

namespace sherpa_onnx {

namespace epprof {

namespace {

struct Event {
  int64_t t_ns;
  int64_t sid;
  int32_t ev;
  int64_t a;
  int64_t b;
  int64_t c;
};

class Profiler {
 public:
  Profiler() {
    const char *path = std::getenv("SHERPA_EP_PROFILE");
    if (!path || !*path) return;
    fp_ = std::fopen(path, "w");
    if (!fp_) {
      std::fprintf(stderr, "SHERPA_EP_PROFILE: cannot open %s\n", path);
      return;
    }
    std::fprintf(fp_, "t_ns\tsid\tevent\ta\tb\tc\n");
    enabled_ = true;
    buf_.reserve(1 << 16);
    writer_ = std::thread([this]() { Loop(); });
  }

  ~Profiler() {
    if (!enabled_) return;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    cv_.notify_all();
    if (writer_.joinable()) writer_.join();
    std::fclose(fp_);
  }

  bool enabled() const { return enabled_; }

  void Push(const Event &e) {
    std::lock_guard<std::mutex> lock(mutex_);
    buf_.push_back(e);
  }

 private:
  void Loop() {
    std::vector<Event> local;
    while (true) {
      bool stop = false;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait_for(lock, std::chrono::milliseconds(100),
                     [this]() { return stop_; });
        local.swap(buf_);
        stop = stop_;
      }
      for (const auto &e : local) {
        std::fprintf(fp_, "%lld\t%lld\t%d\t%lld\t%lld\t%lld\n",
                     static_cast<long long>(e.t_ns),  // NOLINT
                     static_cast<long long>(e.sid), e.ev,  // NOLINT
                     static_cast<long long>(e.a),  // NOLINT
                     static_cast<long long>(e.b),  // NOLINT
                     static_cast<long long>(e.c));  // NOLINT
      }
      local.clear();
      std::fflush(fp_);
      if (stop) break;
    }
  }

  bool enabled_ = false;
  std::FILE *fp_ = nullptr;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::vector<Event> buf_;
  bool stop_ = false;
  std::thread writer_;
};

Profiler &Get() {
  static Profiler p;
  return p;
}

// Force construction (and the env lookup) at static-init time so Enabled()
// is a plain load afterwards.
const bool g_enabled = Get().enabled();

}  // namespace

bool Enabled() { return g_enabled; }

int64_t NowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

void Log(Ev ev, int64_t sid, int64_t a, int64_t b, int64_t c) {
  if (!g_enabled) return;
  Get().Push({NowNs(), sid, static_cast<int32_t>(ev), a, b, c});
}

}  // namespace epprof

}  // namespace sherpa_onnx
