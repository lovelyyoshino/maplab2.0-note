#include "maplab-common/threading-helpers.h"

#include <thread>

#include <gflags/gflags.h>
#include <glog/logging.h>

DEFINE_uint64(
    num_hardware_threads, 0u,
    "Number of hardware threads to announce. (0: use environment variable "
    "MAPLAB_NUM_HARDWARE_THREADS if set, otherwise autodetect)");

namespace common {
namespace internal {

constexpr size_t kDefaultNumHardwareThreads = 4u;

size_t getNumHardwareThreadsImpl() {
  // Check environment variable.
  const char* num_hardware_threads_env_string =
      std::getenv("MAPLAB_NUM_HARDWARE_THREADS");
  if (num_hardware_threads_env_string != nullptr) {
    return std::stoi(num_hardware_threads_env_string);
  }

  const size_t num_detected_threads = std::thread::hardware_concurrency();

  // Fallback to default or user-provided value if the detection failed.
  if (num_detected_threads == 0) {
    static bool warned_once = false;
    if (!warned_once) {
      LOG(WARNING) << "Could not detect the number of hardware threads. "
                   << "Using default of " << kDefaultNumHardwareThreads
                   << ". This can be overridden using the flag "
                   << "num_hardware_threads.";
      warned_once = true;
    }
    return kDefaultNumHardwareThreads;
  }
  return num_detected_threads;
}
}  // namespace internal

size_t getNumHardwareThreads() {
  // Just use the user-provided count if set.
  if (FLAGS_num_hardware_threads > 0) {
    return FLAGS_num_hardware_threads;
  }

  static size_t cached_num_threads = internal::getNumHardwareThreadsImpl();
  return cached_num_threads;
}

}  // namespace common
