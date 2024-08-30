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

    const int Size();
    const std::string_view operator[](const int i);

    std::string data_;
    std::vector<int> wordend_index_;
};