#ifndef REGISTRATION_TOOLBOX_COMMON_SUPPORTED_H_
#define REGISTRATION_TOOLBOX_COMMON_SUPPORTED_H_

#include <map>
#include <string>

namespace regbox {

enum class Aligner : int { PclIcp, PclGIcp, Mock, LpmIcp };

class SupportedAligner {
 public:
  static std::string getKeyForAligner(Aligner aligner);
};

}  // namespace regbox

#endif  // REGISTRATION_TOOLBOX_COMMON_SUPPORTED_H_
