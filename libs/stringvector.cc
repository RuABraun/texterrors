#include "stringvector.h"


StringVector::StringVector(const py::list& words) {
    int total_length = 0;
    for (py::handle obj : words) { 
        const std::string word = obj.cast<std::string>();
        total_length += word.size();
        wordend_index_.push_back(total_length);
    }
    data_.resize(total_length);
    int start_index = 0;
    for (py::handle obj : words) { 
        const std::string word = obj.cast<std::string>();
        std::copy(word.begin(), word.end(), data_.begin() + start_index);
        start_index += word.size();
    }
    current_index_ = 0;
}

StringVector::StringVector(const vector<std::string>& words) {
    int total_length = 0;
    for (std::string word : words) { 
        total_length += word.size();
        wordend_index_.push_back(total_length);
    }
    current_index_ = 0;
}

const int StringVector::size() const {
    return wordend_index_.size();
}

const std::string_view StringVector::operator[](const int i) const {
    if (i < 0 || i >= size()) {
        throw std::runtime_error("Invalid index");
    }
    int start_index = 0;
    if (i > 0) {
        start_index = wordend_index_[i-1];
    }
    int length = wordend_index_[i] - start_index;
    return std::string_view(data_).substr(start_index, length);
}

StringVector StringVector::iter() {
    current_index_ = 0;
    return *this;
}

const std::string_view StringVector::next() {
    if (current_index_ == size()) {
        throw pybind11::stop_iteration();
    }
    return (*this)[current_index_++];
}

std::string StringVector::Str() const {
    std::string repr = "";
    for (int i = 0; i < size(); i++) {
        repr += std::string{(*this)[i]} + " ";
    }
    return repr;
}

StringVector::~StringVector() {}


void init_stringvector(py::module &m) {
    py::class_<StringVector>(m, "StringVector")
        .def(py::init<const py::list&>())
        .def("size", &StringVector::size)
        .def("__len__", &StringVector::size)
        .def("__getitem__", &StringVector::operator[])
        .def("__iter__", &StringVector::iter)
        .def("__next__", &StringVector::next)
        .def("__str__", &StringVector::Str);
}