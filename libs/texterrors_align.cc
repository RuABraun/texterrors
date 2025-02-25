#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <queue>
#include <iostream>
#include <math.h>
#include "stringvector.h"


namespace py = pybind11;

typedef int32_t int32;


bool isclose(double a, double b) {
  return abs(a - b) < 0.0001;
}

struct Pair {
    Pair() {}
    Pair(int16_t f_, int16_t s_) {
        i = f_;
        j = s_;
    }
    int16_t i;
    int16_t j;
};


int calc_edit_distance_fast(int32* cost_mat, const char* a, const char* b,
                     const int32 M, const int32 N) {
  int row_length = N+1;
  // std::cout << "STARTING M="<< M<< " N="<<N<<std::endl;
  for (int32 i = 0; i <= M; ++i) {
    for (int32 j = 0; j <= N; ++j) {

      if (i == 0 && j == 0) {
        cost_mat[0] = 0;
        continue;
      }
      if (i == 0) {
        cost_mat[j] = cost_mat[j - 1] + 1;
        continue;
      }
      if (j == 0) {
        cost_mat[row_length] = cost_mat[0] + 1;
        continue;
      }
      int32 transition_cost = a[i-1] == b[j-1] ? 0 : 1;

      int32 upc = cost_mat[j] + 1;
      int32 leftc = cost_mat[row_length + j - 1] + 1;
      int32 diagc = cost_mat[j - 1] + transition_cost;
      int32 cost = std::min(upc, std::min(leftc, diagc) );

      cost_mat[row_length + j] = cost;
      cost_mat[j - 1] = cost_mat[row_length + j - 1];  // copying result up after use
    }
    if (i > 0) {
      cost_mat[N] = cost_mat[row_length + N];
    }

    // std::cout << "row "<<i;
    // for (int32 j = 0; j <= N; ++j) {
    //   std::cout << " "<<cost_mat[j];
    // }
    // std::cout << std::endl;
  }
  // std::cout << "last row";
  // for (int32 j = 0; j <= N; ++j) {
  //   std::cout <<" "<<cost_mat[row_length + j];
  // }
  // std::cout << std::endl;

  return cost_mat[row_length - 1];
}


template <class T>
void create_lev_cost_mat(int32* cost_mat, const T* a, const T* b,
                     const int32 M, const int32 N) {
  int row_length = N+1;
  for (int32 i = 0; i <= M; ++i) {
    for (int32 j = 0; j <= N; ++j) {

      if (i == 0 && j == 0) {
        cost_mat[0] = 0;
        continue;
      }
      if (i == 0) {
        int new_value = cost_mat[j - 1] + 3;
        cost_mat[j] = new_value;
        continue;
      }
      if (j == 0) {
        int new_value = cost_mat[(i-1) * row_length] + 3;
        cost_mat[i * row_length] = new_value;
        continue;
      }
      int32 transition_cost = a[i-1] == b[j-1] ? 0 : 1;

      int32 upc = cost_mat[(i-1) * row_length + j] + 3;
      int32 leftc = cost_mat[i * row_length + j - 1] + 3;
      int32 diagc = cost_mat[(i-1) * row_length + j - 1] + 4 * transition_cost;
      int32 cost = std::min(upc, std::min(leftc, diagc) );
      cost_mat[i * row_length + j] = cost;
    }
  }
}

template <class T>
int levdistance(const T* a, const T* b, int32 M, int32 N) {
  if (!M) return N;
  if (!N) return M;
  std::vector<int32> cost_mat((M+1)*(N+1));
  create_lev_cost_mat(cost_mat.data(), a, b, M, N);
  int cost = 0;
  int i = M, j = N;
  int row_length = N+1;
  while (i != 0 || j != 0) {
    if (i == 0) {
      j--;
      cost++;
    } else if (j == 0) {
      i--;
      cost++;
    } else {
      int current_cost = cost_mat[i * row_length + j];
      int diagc = cost_mat[(i-1) * row_length + j - 1];
      int upc = cost_mat[(i-1) * row_length + j];
      int leftc = cost_mat[i * row_length + j - 1];
      int32 transition_cost = a[i-1] == b[j-1] ? 0 : 1;
      if (diagc + 4 * transition_cost == current_cost) {
        i--, j--;
        if (current_cost != diagc) cost++;
      } else if (upc + 3 == current_cost) {
        i--;
        cost++;
      } else if (leftc + 3 == current_cost) {
        j--;
        cost++;
      } else {
        std::cerr <<diagc<<" "<<upc<<" "<<leftc<<" "<<current_cost<<" "<< transition_cost<< " WTF"<<std::endl;
        throw "Should not happen!";
      }
    }
//    std::cout << cost << " "<<i<<" "<<j<<std::endl;
  }
  return cost;
}

template <class T>
int lev_distance(std::vector<T> a, std::vector<T> b) {
  return levdistance(a.data(), b.data(), a.size(), b.size());
}

int lev_distance_str(std::string a, std::string b) {
  return levdistance(a.data(), b.data(), a.size(), b.size());
}

int calc_edit_distance_fast_str(std::string a, std::string b) {
  std::vector<int> buffer(a.size() + b.size() + 2);
  return calc_edit_distance_fast(buffer.data(), a.data(), b.data(), a.size(), b.size());
}

enum direction{diag, move_left, up};

std::vector<std::tuple<int32, int32> > get_best_path(py::array_t<double> array, 
                  const StringVector& words_a,
                  const StringVector& words_b, const bool use_chardiff, const bool use_fast_edit_distance=true) {
  auto buf = array.request();
  double* cost_mat = (double*) buf.ptr;
  int32_t numr = array.shape()[0], numc = array.shape()[1];
  std::vector<int32> char_dist_buffer;
  if (use_chardiff) {
    char_dist_buffer.resize(100);
  }

  std::vector<std::tuple<int, int> > bestpath;
  int i = numr - 1, j = numc - 1;
  while (i != 0 || j != 0) {
    double upc, leftc, diagc;
    direction direc;
    if (i == 0) {
      direc = move_left;
    } else if (j == 0) {
      direc = up;
    } else {
      float current_cost = cost_mat[i * numc + j];
      upc = cost_mat[(i-1) * numc + j];
      leftc = cost_mat[i * numc + j - 1];
      diagc = cost_mat[(i-1) * numc + j - 1];
      const std::string_view a = words_a[i-1];
      const std::string_view b = words_b[j-1];
      double up_trans_cost = 1.0;
      double left_trans_cost = 1.0;
      double diag_trans_cost;
      if (use_chardiff) {
        int alen = a.size();
        int blen = b.size();
        if (alen >= 50 || blen >= 50) {
          throw std::runtime_error("Word is too long! Increase buffer");
        }
        if (use_fast_edit_distance) {
          diag_trans_cost =
          calc_edit_distance_fast(char_dist_buffer.data(), a.data(), b.data(), a.size(), b.size()) / static_cast<double>(std::max(a.size(), b.size())) * 1.5;
        } else {
          diag_trans_cost =
          levdistance(a.data(), b.data(), a.size(), b.size()) / static_cast<double>(std::max(a.size(), b.size())) * 1.5;
        }
      } else {
        diag_trans_cost = a == b ? 0. : 1.;
      }

      if (isclose(diagc + diag_trans_cost, current_cost)) {
        direc = diag;
      } else if (isclose(upc + up_trans_cost, current_cost)) {
        direc = up;
      } else if (isclose(leftc + left_trans_cost, current_cost)) {
        direc = move_left;
      } else {
        std::cout << a <<" "<<b<<" "<<i<<" "<<j<<" trans "<<diag_trans_cost<<" "<<left_trans_cost<<" "<<up_trans_cost<<" costs "<<current_cost<<" "<<diagc<<" "<<leftc<<" "<<upc <<std::endl;
        std::cout << (diag_trans_cost + diagc == current_cost) <<std::endl;
        std::cout << diag_trans_cost + diagc <<" "<<current_cost <<std::endl;
        throw std::runtime_error("Should not be possible !");
      }
    }

    if (direc == up) {
      i--;
      bestpath.emplace_back(i, -1);  // -1 means null token
    } else if (direc == move_left) {
      j--;
      bestpath.emplace_back(-1, j);
    } else if (direc == diag) {
      i--, j--;
      bestpath.emplace_back(i, j);
    }
  }
  return bestpath;
}


std::vector<std::tuple<int32, int32> > get_best_path_lists(py::array_t<double> array, 
                  const std::vector<std::string>& words_a,
                  const std::vector<std::string>& words_b, const bool use_chardiff, const bool use_fast_edit_distance=true) {
  auto buf = array.request();
  double* cost_mat = (double*) buf.ptr;
  int32_t numr = array.shape()[0], numc = array.shape()[1];
  std::vector<int32> char_dist_buffer;
  if (use_chardiff) {
    char_dist_buffer.resize(100);
  }

  std::vector<std::tuple<int, int> > bestpath;
  int i = numr - 1, j = numc - 1;
  while (i != 0 || j != 0) {
    double upc, leftc, diagc;
    direction direc;
    if (i == 0) {
      direc = move_left;
    } else if (j == 0) {
      direc = up;
    } else {
      float current_cost = cost_mat[i * numc + j];
      upc = cost_mat[(i-1) * numc + j];
      leftc = cost_mat[i * numc + j - 1];
      diagc = cost_mat[(i-1) * numc + j - 1];
      const std::string& a = words_a[i-1];
      const std::string& b = words_b[j-1];
      double up_trans_cost = 1.0;
      double left_trans_cost = 1.0;
      double diag_trans_cost;
      if (use_chardiff) {
        int alen = a.size();
        int blen = b.size();
        if (alen >= 50 || blen >= 50) {
          throw std::runtime_error("Word is too long! Increase buffer");
        }
        if (use_fast_edit_distance) {
          diag_trans_cost =
          calc_edit_distance_fast(char_dist_buffer.data(), a.data(), b.data(), a.size(), b.size()) / static_cast<double>(std::max(a.size(), b.size())) * 1.5;
        } else {
          diag_trans_cost =
          levdistance(a.data(), b.data(), a.size(), b.size()) / static_cast<double>(std::max(a.size(), b.size())) * 1.5;
        }
      } else {
        diag_trans_cost = a == b ? 0. : 1.;
      }

      if (isclose(diagc + diag_trans_cost, current_cost)) {
        direc = diag;
      } else if (isclose(upc + up_trans_cost, current_cost)) {
        direc = up;
      } else if (isclose(leftc + left_trans_cost, current_cost)) {
        direc = move_left;
      } else {
        std::cout << a <<" "<<b<<" "<<i<<" "<<j<<" trans "<<diag_trans_cost<<" "<<left_trans_cost<<" "<<up_trans_cost<<" costs "<<current_cost<<" "<<diagc<<" "<<leftc<<" "<<upc <<std::endl;
        std::cout << (diag_trans_cost + diagc == current_cost) <<std::endl;
        std::cout << diag_trans_cost + diagc <<" "<<current_cost <<std::endl;
        throw std::runtime_error("Should not be possible !");
      }
    }

    if (direc == up) {
      i--;
      bestpath.emplace_back(i, -1);  // -1 means null token
    } else if (direc == move_left) {
      j--;
      bestpath.emplace_back(-1, j);
    } else if (direc == diag) {
      i--, j--;
      bestpath.emplace_back(i, j);
    }
  }
  return bestpath;
}

void get_best_path_ctm(py::array_t<double> array, py::list& bestpath_lst, std::vector<std::string> texta,
                   std::vector<std::string> textb, std::vector<double> times_a, std::vector<double> times_b,
                   std::vector<double> durs_a, std::vector<double> durs_b) {
  auto buf = array.request();
  double* cost_mat = (double*) buf.ptr;
  int32_t numr = array.shape()[0], numc = array.shape()[1];

  if (numr > 32000 || numc > 32000) throw std::runtime_error("Input sequences are too large!");

  std::vector<Pair> bestpath;
  int i = numr - 1, j = numc - 1;
  bestpath.emplace_back(i, j);
  while (i != 0 || j != 0) {
    double upc, leftc, diagc;
    int idx;  // 0 up, 1 left, 2 diagonal
    if (i == 0) {
      idx = 1;
    } else if (j == 0) {
      idx = 0;
    } else {
      float current_cost = cost_mat[i * numc + j];
      upc = cost_mat[(i-1) * numc + j];
      leftc = cost_mat[i * numc + j - 1];
      diagc = cost_mat[(i-1) * numc + j - 1];

      double time_cost;
      if (i == 0 || j == 0) {
        time_cost = 0.;
      } else {
        double start_a = times_a[i - 1];
        double start_b = times_b[j - 1];
        double end_a = start_a + durs_a[i - 1];
        double end_b = start_b + durs_b[j - 1];
        double overlap;
        if (start_a > end_b) {
          overlap = end_b - start_a;
        } else if (start_b > end_a) {
          overlap = end_a - start_b;
        } else if (start_a > start_b) {
          double min_end = std::min(end_a, end_b); 
          overlap = min_end - start_a;
        } else {
          double min_end = std::min(end_a, end_b); 
          overlap = min_end - start_b;
        }
        time_cost = -overlap;
      }
      
      double up_trans_cost = 1. + time_cost;
      double left_trans_cost = 1. + time_cost;
      double diag_trans_cost = texta[i] == textb[j] ? 0. + time_cost : 1. + time_cost;

      if (isclose(upc + up_trans_cost, current_cost)) {
        idx = 0;
      } else if (isclose(leftc + left_trans_cost, current_cost)) {
        idx = 1;
      } else if (isclose(diagc + diag_trans_cost, current_cost)) {
        idx = 2;
      } else {
        std::cout << texta[i] <<" "<<textb[j]<<" "<<i<<" "<<j<<" trans "<<diag_trans_cost<<" "<<left_trans_cost<<" "<<up_trans_cost<<" costs "<<current_cost<<" "<<diagc<<" "<<leftc<<" "<<upc <<" times " << times_a[i] << " "<<times_b[j]<<std::endl;
        std::cout << (diag_trans_cost + diagc == current_cost) <<std::endl;
        std::cout << diag_trans_cost + diagc <<" "<<current_cost <<std::endl;
        throw std::runtime_error("Should not be possible !");
      }
    }

    if (idx == 0) {
      i--;
    } else if (idx == 1) {
      j--;
    } else if (idx == 2) {
      i--, j--;
    }
    bestpath.emplace_back(i, j);
  }

  if (bestpath.size() == 1) throw std::runtime_error("No best path found!");
  for (int32_t k = 0; k < bestpath.size(); k++) {
    bestpath_lst.append(bestpath[k].i);
    bestpath_lst.append(bestpath[k].j);
  }
}



int calc_sum_cost(py::array_t<double> array, const StringVector& words_a,
                  const StringVector& words_b, const bool use_chardist, const bool use_fast_edit_distance=true) {
  if ( array.ndim() != 2 )
    throw std::runtime_error("Input should be 2-D NumPy array");

  int M1 = array.shape()[0], N1 = array.shape()[1];
  if (M1 - 1 != words_a.size() || N1 - 1 != words_b.size()) throw std::runtime_error("Sizes do not match!");
  auto buf = array.request();
  double* ptr = (double*) buf.ptr;

  std::vector<int32> char_dist_buffer;
  if (use_chardist) {
    char_dist_buffer.resize(100);
  }

  ptr[0] = 0;
  for (int32 i = 1; i < M1; i++) ptr[i*N1] = ptr[(i-1)*N1] + 1;
  for (int32 j = 1; j < N1; j++) ptr[j] = ptr[j-1] + 1;
  for(int32 i = 1; i < M1; i++) {
    for(int32 j = 1; j < N1; j++) {
      double transition_cost;
      if (use_chardist) {
        const std::string_view a = words_a[i-1];
        const std::string_view b = words_b[j-1];
        int alen = a.size();
        int blen = b.size();
        if (alen >= 50 || blen >= 50) {
          throw std::runtime_error("Word is too long! Increase buffer");
        }
        if (use_fast_edit_distance) {
          transition_cost =
          calc_edit_distance_fast(char_dist_buffer.data(), a.data(), b.data(), a.size(), b.size()) / static_cast<double>(std::max(a.size(), b.size())) * 1.5;
        } else {
          transition_cost =
          levdistance(a.data(), b.data(), a.size(), b.size()) / static_cast<double>(std::max(a.size(), b.size())) * 1.5;
        }
      } else {
        transition_cost = words_a[i-1] == words_b[j-1] ? 0. : 1.;
      }

      double upc = ptr[(i-1) * N1 + j] + 1.;
      double leftc = ptr[i * N1 + j - 1] + 1.;
      double diagc = ptr[(i-1) * N1 + j - 1] + transition_cost;
      double sum = std::min(upc, std::min(leftc, diagc));
      ptr[i * N1 + j] = sum;
    }
  }
  return ptr[M1*N1 - 1];
}



int calc_sum_cost_lists(py::array_t<double> array, const std::vector<std::string>& words_a,
                  const std::vector<std::string>& words_b, const bool use_chardist, const bool use_fast_edit_distance=true) {
  if ( array.ndim() != 2 )
    throw std::runtime_error("Input should be 2-D NumPy array");

  int M1 = array.shape()[0], N1 = array.shape()[1];
  if (M1 - 1 != words_a.size() || N1 - 1 != words_b.size()) throw std::runtime_error("Sizes do not match!");
  auto buf = array.request();
  double* ptr = (double*) buf.ptr;

  std::vector<int32> char_dist_buffer;
  if (use_chardist) {
    char_dist_buffer.resize(100);
  }

  ptr[0] = 0;
  for (int32 i = 1; i < M1; i++) ptr[i*N1] = ptr[(i-1)*N1] + 1;
  for (int32 j = 1; j < N1; j++) ptr[j] = ptr[j-1] + 1;
  for(int32 i = 1; i < M1; i++) {
    for(int32 j = 1; j < N1; j++) {
      double transition_cost;
      if (use_chardist) {
        const std::string& a = words_a[i-1];
        const std::string& b = words_b[j-1];
        int alen = a.size();
        int blen = b.size();
        if (alen >= 50 || blen >= 50) {
          throw std::runtime_error("Word is too long! Increase buffer");
        }
        if (use_fast_edit_distance) {
          transition_cost =
          calc_edit_distance_fast(char_dist_buffer.data(), a.data(), b.data(), a.size(), b.size()) / static_cast<double>(std::max(a.size(), b.size())) * 1.5;
        } else {
          transition_cost =
          levdistance(a.data(), b.data(), a.size(), b.size()) / static_cast<double>(std::max(a.size(), b.size())) * 1.5;
        }
      } else {
        transition_cost = words_a[i-1] == words_b[j-1] ? 0. : 1.;
      }

      double upc = ptr[(i-1) * N1 + j] + 1.;
      double leftc = ptr[i * N1 + j - 1] + 1.;
      double diagc = ptr[(i-1) * N1 + j - 1] + transition_cost;
      double sum = std::min(upc, std::min(leftc, diagc));
      ptr[i * N1 + j] = sum;
    }
  }
  return ptr[M1*N1 - 1];
}


int calc_sum_cost_ctm(py::array_t<double> array, std::vector<std::string>& texta,
                      std::vector<std::string>& textb, std::vector<double> times_a, std::vector<double> times_b,
                      std::vector<double> durs_a, std::vector<double> durs_b) {
  if ( array.ndim() != 2 )
    throw std::runtime_error("Input should be 2-D NumPy array");

  int M = array.shape()[0], N = array.shape()[1];
  if (M != texta.size() || N != textb.size()) throw std::runtime_error("  s do not match!");
  auto buf = array.request();
  double* ptr = (double*) buf.ptr;
//  std::cout << "STARTING"<<std::endl;
  for(int32 i = 0; i < M; i++) {
    for(int32 j = 0; j < N; j++) {
      double transition_cost, a_cost, b_cost;
      double time_cost;
      if (i == 0 || j == 0) {
        time_cost = 0.;
      } else {
        double start_a = times_a[i - 1];
        double start_b = times_b[j - 1];
        double end_a = start_a + durs_a[i - 1];
        double end_b = start_b + durs_b[j - 1];
        double overlap;
        if (start_a > end_b) {
          overlap = end_b - start_a;
        } else if (start_b > end_a) {
          overlap = end_a - start_b;
        } else if (start_a > start_b) {
          double min_end = std::min(end_a, end_b); 
          overlap = min_end - start_a;
        } else {
          double min_end = std::min(end_a, end_b); 
          overlap = min_end - start_b;
        }
        time_cost = -overlap;
      }
      
      a_cost = 1. + time_cost;
      b_cost = 1. + time_cost;
      transition_cost = texta[i] == textb[j] ? 0. + time_cost : 1. + time_cost;

      if (i == 0 && j == 0) {
        ptr[0] = 0;
        continue;
      }
      if (i == 0)  {
        ptr[j] = ptr[j - 1] + b_cost;
        continue;
      }
      if (j == 0) {
        ptr[i * N] = ptr[(i-1) * N] + a_cost;
        continue;
      }

      double upc = ptr[(i-1) * N + j] + a_cost;
      double leftc = ptr[i * N + j - 1] + b_cost;
      double diagc = ptr[(i-1) * N + j - 1] + transition_cost;
      double sum = std::min(upc, std::min(leftc, diagc));
      ptr[i * N + j] = sum;
    }
  }
  return ptr[(M-1) * N + N - 1];
}


void init_stringvector(py::module_ &m);


PYBIND11_MODULE(texterrors_align,m) {
  m.doc() = "pybind11 plugin";
  m.def("calc_sum_cost", &calc_sum_cost, "Calculate summed cost matrix");
  m.def("calc_sum_cost_lists", &calc_sum_cost_lists, "Calculate summed cost matrix");
  m.def("calc_sum_cost_ctm", &calc_sum_cost_ctm, "Calculate summed cost matrix");
  m.def("get_best_path", &get_best_path, "get_best_path");
  m.def("get_best_path_ctm", &get_best_path_ctm, "get_best_path_ctm");
  m.def("get_best_path_lists", &get_best_path_lists, "get_best_path_lists");
  m.def("lev_distance", lev_distance<int>);
  m.def("lev_distance", lev_distance<char>);
  m.def("lev_distance_str", &lev_distance_str);
  m.def("calc_edit_distance_fast_str", &calc_edit_distance_fast_str);
  init_stringvector(m);
}
