#include "ceres-error-terms/ceres-signal-handler.h"

#include <ceres/types.h>
#include <signal.h>
#include <vector>

namespace ceres_error_terms {

std::vector<SignalHandlerCallback*> signal_handler_links;

SignalHandlerCallback::SignalHandlerCallback()
    : terminate_after_next_iteration_(false) {
  // 安装一个临时的SIGINT处理程序来中止当前的优化
  signal_handler_links.push_back(
      this);  // 将当前的SignalHandlerCallback对象添加到signal_handler_links中
  sigaction(
      SIGINT, nullptr, &previous_signal_handler_);  // 获取之前的SIGINT处理程序
  signal(
      SIGINT,
      static_cast<sig_t>(&signalHandler));  // 安装一个临时的SIGINT处理程序
}

SignalHandlerCallback::~SignalHandlerCallback() {
  // 恢复原来的信号处理程序
  signal(
      SIGINT, previous_signal_handler_.sa_handler);  // 恢复之前的SIGINT处理程序

  // 从全局处理程序链接中删除指向此信号处理程序的指针
  auto it = signal_handler_links.begin();  // 获取signal_handler_links的迭代器
  while (it != signal_handler_links.end()) {
    if (*it == this) {
      it = signal_handler_links.erase(it);
    } else {
      ++it;
    }
  }
}

void SignalHandlerCallback::signalHandler(int signal) {
  CHECK_EQ(signal, SIGINT);  // 检查signal是否为SIGINT
  CHECK(!signal_handler_links.empty())
      << "Signal was received, but no signal handler was registered!";

  for (SignalHandlerCallback* callback : signal_handler_links) {
    callback->terminate_after_next_iteration_ =
        true;  // 将terminate_after_next_iteration_设置为true
  }
  LOG(WARNING)
      << "User requested optimization termination after next update...";
}

ceres::CallbackReturnType SignalHandlerCallback::operator()(
    const ceres::IterationSummary&) {
  if (terminate_after_next_iteration_) {
    terminate_after_next_iteration_ =
        false;  // 将terminate_after_next_iteration_设置为false
    return ceres::
        SOLVER_TERMINATE_SUCCESSFULLY;  // 返回SOLVER_TERMINATE_SUCCESSFULLY
  }
  return ceres::SOLVER_CONTINUE;  // 返回SOLVER_CONTINUE
}

}  // namespace ceres_error_terms
