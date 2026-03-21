#include <string>
#include <string_view>
#include <vector>
#include <nanobind/nanobind.h>
#include <stdexcept>

namespace nb = nanobind;

using namespace std;


class StringVector {
public:
    StringVector(const nb::list& words);
    StringVector(const vector<std::string>& words);
    ~StringVector();

    const int size() const;
    const std::string_view operator[](const int i) const;
    StringVector iter();
    const std::string_view next();
    std::string Str() const;

    std::string data_;
    std::vector<int> wordend_index_;
    int current_index_;
};
