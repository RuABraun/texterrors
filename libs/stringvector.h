#include <string>
#include <string_view>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>

namespace py = pybind11;

using namespace std;


class StringVector {
public:
    StringVector(const py::list& words);
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