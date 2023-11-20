#ifndef NEMO_BRAIN_UTIL_H_
#define NEMO_BRAIN_UTIL_H_

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <iterator>
#include <vector>

namespace nemo {

inline size_t NumCommon(const std::vector<uint32_t>& a_in,
                        const std::vector<uint32_t>& b_in) {
  std::vector<uint32_t> a = a_in;
  std::vector<uint32_t> b = b_in;
  std::sort(a.begin(), a.end());
  std::sort(b.begin(), b.end());
  std::vector<uint32_t> c;
  std::set_intersection(a.begin(), a.end(), b.begin(), b.end(),
                        std::back_inserter(c));
  return c.size();
}

}  // namespace nemo
#endif  // NEMO_BRAIN_UTIL_H_
