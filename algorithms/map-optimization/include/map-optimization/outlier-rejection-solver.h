#ifndef MAP_OPTIMIZATION_OUTLIER_REJECTION_SOLVER_H_
#define MAP_OPTIMIZATION_OUTLIER_REJECTION_SOLVER_H_

#include <iomanip>
#include <memory>
#include <string>
#include <vector>

#include <map-optimization/optimization-problem.h>
#include <maplab-common/stringprintf.h>

DECLARE_bool(ba_hide_iterations_console_output);
namespace map_optimization {

struct OutlierRejectionSolverOptions {
  static OutlierRejectionSolverOptions initFromFlags();

  // Run an outlier rejection loop every n iterations. Landmarks behind the
  // camera will be removed from the problem and flagged as kBad. Optionally,
  // landmarks are removed based on the reprojection errors.
  int reject_outliers_every_n_iters = 3;

  // Classify and remove landmarks based on reprojection errors.
  bool use_reprojection_error = false;
  // Reprojection error threshold for landmarks based on observations
  double max_reprojection_error_px = 50;
};

class OutlierRejectionCallback : public ceres::IterationCallback {
 public:
  explicit OutlierRejectionCallback(double initial_trust_region_radius)
      : iteration_(0),
        initial_trust_region_radius_(initial_trust_region_radius) {}

  virtual ~OutlierRejectionCallback() {}

  ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary) {
    iteration_summaries_.emplace_back(summary);

    if (iteration_ == 0) {
      VLOG_IF(0, !FLAGS_ba_hide_iterations_console_output)
          << "iter      cost      cost_change  |gradient|   |step|    tr_ratio "
             " tr_radius  ls_iter  iter_time  total_time";  // NOLINT
    }

    if (!(iteration_ > 0 && summary.iteration == 0)) {
      const char* kReportRowFormat =
          "% 4d % 8e   % 3.2e   % 3.2e  % 3.2e  % 3.2e % 3.2e     % 4d   % "
          "3.2e   % 3.2e";  // NOLINT

      std::string output = common::StringPrintf(
          kReportRowFormat, iteration_, summary.cost, summary.cost_change,
          summary.gradient_max_norm, summary.step_norm,
          summary.relative_decrease, summary.trust_region_radius,
          summary.linear_solver_iterations, summary.iteration_time_in_seconds,
          summary.cumulative_time_in_seconds);
      VLOG_IF(0, !FLAGS_ba_hide_iterations_console_output) << output;

      ++iteration_;
    }
    return ceres::SOLVER_CONTINUE;
  }

  int iteration_;
  double initial_trust_region_radius_;

  std::vector<ceres::IterationSummary> iteration_summaries_;
};

ceres::TerminationType solveWithOutlierRejection(
    const ceres::Solver::Options& solver_options,
    const OutlierRejectionSolverOptions& rejection_options,
    OptimizationProblem* optimization_problem,
    OptimizationProblemResult* result);

ceres::TerminationType solveWithOutlierRejection(
    const ceres::Solver::Options& solver_options,
    const OutlierRejectionSolverOptions& rejection_options,
    OptimizationProblem* optimization_problem);

}  // namespace map_optimization
#endif  // MAP_OPTIMIZATION_OUTLIER_REJECTION_SOLVER_H_
