#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <queue>
#include <iostream>


namespace py = pybind11;

typedef int32_t int32;

struct Pair {
		Pair() {}
    Pair(int16_t f_, int16_t s_) {
        i = f_;
        j = s_;
    }
    int16_t i;
    int16_t j;
};

template <class T>
void create_cost_mat(int32* cost_mat, const T* a, const T* b,
                     const int32 M, const int32 N) {
  int row_length = N+1;
//  std::string s = "";
  for (int32 i = 0; i <= M; ++i) {
    for (int32 j = 0; j <= N; ++j) {

      if (i == 0 && j == 0) {
        cost_mat[0] = 0;
//        s += std::to_string(0) + " ";
        continue;
      }
      if (i == 0) {
        int new_value = cost_mat[j - 1] + 3;
        cost_mat[j] = new_value;
//        s += std::to_string(new_value) + " ";
        continue;
      }
      if (j == 0) {
        int new_value = cost_mat[(i-1) * row_length] + 3;
        cost_mat[i * row_length] = new_value;
//        s += std::to_string(new_value) + " ";
        continue;
      }
      int32 transition_cost = a[i-1] == b[j-1] ? 0 : 1;

      int32 upc = cost_mat[(i-1) * row_length + j] + 3;
      int32 leftc = cost_mat[i * row_length + j - 1] + 3;
      int32 diagc = cost_mat[(i-1) * row_length + j - 1] + 4 * transition_cost;
      int32 cost = std::min(upc, std::min(leftc, diagc) );
//      s += std::to_string(cost) + " ";
      cost_mat[i * row_length + j] = cost;
    }
//    s += "\n";
//    std::string ss = "";
//    for (int k = 0; k < (M+1)*(N+1); k++) {
//      ss += std::to_string(cost_mat[k]) + " ";
//    }
//    std::cout << ss << std::endl;
  }
//  std::cout <<s<<std::endl;
}

template <class T>
int levdistance(const T* a, const T* b, int32 M, int32 N) {
  if (!M) return N;
  if (!N) return M;
  std::vector<int32> cost_mat((M+1)*(N+1));
  create_cost_mat(cost_mat.data(), a, b, M, N);
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
      if (diagc <= upc && diagc <= leftc) {
        i--, j--;
        if (current_cost != diagc) cost++;
      } else if (upc < diagc && upc <= leftc) {
        i--;
        if (current_cost != upc) cost++;
      } else if (leftc < diagc && leftc <= upc) {
        j--;
        if (current_cost != leftc) cost++;
      } else {
        std::cerr <<diagc<<" "<<upc<<" "<<leftc<< " WTF"<<std::endl;
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


void get_best_path(py::array_t<int32_t> array, py::list& bestpath_lst, std::vector<int32_t>& texta,
									 std::vector<int32_t>& textb) {
	auto buf = array.request();
	int32_t* cost_mat = (int32_t*) buf.ptr;
	int32_t numr = array.shape()[0], numc = array.shape()[1];
	int32_t maxlen = numr + numc;

	if (numr > 32000 || numc > 32000) throw std::runtime_error("Input sequences are too large!");

	std::queue< std::vector<Pair>> paths_to_explore;
	std::vector<Pair> bestpath;
	std::vector<Pair> path;
	int32_t best_continuous_match_len = -1;
	path.reserve(maxlen);
	int i = numr - 1, j = numc - 1;
	path.push_back(Pair(i, j));
	int32_t max_paths_explore = 30000;
	int32_t paths_found = 0;
//	std::string a = "";
//	for (int n = 0; n < texta.size(); n++) {
//	  a += std::to_string(texta[n]) + " ";
//	}
//  std::string b = "";
//  for (int n = 0; n < textb.size(); n++) {
//    b += std::to_string(textb[n]) + " ";
//  }
//  std::cout << "A "<<a<<"\n";
//  std::cout << "B "<<b<<"\n";
	while (true) {
		if (i == 0 && j == 0) {
			int32_t path_len = path.size();
			int32_t startidx = -1, endidx = -1;
      int32_t continuous_match_len = 0;
//      std::string str_pairs = "pairs ";
			for(int32_t n = 0; n < path_len - 1; n++) {
				Pair& pair = path[n];
//        str_pairs += std::to_string(texta[path[n].i])+","+std::to_string(textb[path[n].j])+ " ";
				if (pair.i - 1 == path[n+1].i && pair.j - 1 == path[n+1].j && texta[path[n].i] == textb[path[n].j]) {
          continuous_match_len++;
				}
			}
//      std::cout << str_pairs << std::endl;
			if (bestpath.size() == 0 || continuous_match_len > best_continuous_match_len) {
				best_continuous_match_len = continuous_match_len;
				bestpath = path;
//				std::cout << "bestpath\n";
			}
//			std::string s = "";
//			for (int n = 0; n < path_len; n++) {
//        Pair& pair = path[n];
//        s += std::to_string(pair.i) + "," + std::to_string(pair.j)+" ";
//			}
//			std::cout << "Path "<<continuous_match_len<<" " << s <<std::endl;
			if (paths_to_explore.empty()) {
				break;
			}
			path = paths_to_explore.front();
			Pair& p = path.back();
			i = p.i, j = p.j;
			paths_to_explore.pop();
		}

		int32_t upc, leftc, diagc;
		int idx;  // 0 up, 1 left, 2 diagonal
		if (i == 0) {
		  idx = 1;
		} else if (j == 0) {
		  idx = 0;
		} else {
      upc = cost_mat[(i-1) * numc + j];
      leftc = cost_mat[i * numc + j - 1];
      diagc = cost_mat[(i-1) * numc + j - 1];

      if (diagc < leftc && diagc < upc) {
        idx = 2;
      } else if (leftc < upc && leftc < diagc) {
        idx = 1;
      } else if (upc < leftc && upc < diagc) {
        idx = 0;
      } else {
        if (paths_found < max_paths_explore) {
          if (upc == diagc && upc < leftc) {
            std::vector<Pair> pathcopied(path);
            Pair explorep(i - 1, j);
            pathcopied.push_back(explorep);
            paths_to_explore.push(pathcopied);
            paths_found++;
            idx = 2;
          } else if (leftc == diagc && leftc < upc) {
            std::vector<Pair> pathcopied(path);
            Pair explorep(i, j - 1);
            pathcopied.push_back(explorep);
            paths_to_explore.push(pathcopied);
            paths_found++;
            idx = 2;
          } else if (leftc == diagc && leftc == upc) {
            idx = 2;
//            throw std::runtime_error("Should not be possible !");
          }
//          std::cout << paths_to_explore.size()<<std::endl;
        }
      }
		}

    if (idx == 0) {
      i--;
    } else if (idx == 1) {
      j--;
    } else if (idx == 2) {
      i--, j--;
    } else {
      throw "WTF";
    }
    path.emplace_back(i, j);
	}

	if (bestpath.size() == 1) throw std::runtime_error("No best path found!");
	for(int32_t k = 0; k < bestpath.size(); k++) {
		bestpath_lst.append(bestpath[k].i);
		bestpath_lst.append(bestpath[k].j);
	}
}


int calc_sum_cost(py::array_t<int32_t> array, std::vector<std::string>& texta,
                         std::vector<std::string>& textb, bool use_chardist) {
	if ( array.ndim() != 2 )
    throw std::runtime_error("Input should be 2-D NumPy array");

  int M = array.shape()[0], N = array.shape()[1];
  if (M != texta.size() || N != textb.size()) throw std::runtime_error("Sizes do not match!");
  auto buf = array.request();
  int32_t* ptr = (int32_t*) buf.ptr;
//  std::cout << "STARTING"<<std::endl;
  for(int32 i = 0; i < M; i++) {
		for(int32 j = 0; j < N; j++) {
      int32 transition_cost;
      int32 a_cost, b_cost;
		  if (use_chardist) {
		    std::string& a = texta[i];
        std::string& b = textb[j];
        transition_cost = levdistance(a.data(), b.data(), a.size(), b.size());
        a_cost = a.size();
        b_cost = b.size();
      } else {
        a_cost = 1;
        b_cost = 1;
        transition_cost = texta[i] == textb[j] ? 0 : 1;
		  }

		  if (i == 0 && j == 0) {
		    ptr[0] = 0;
        continue;
		  }
		  if (i == 0)  {
		    ptr[j] = ptr[j - 1] + 3;
        continue;
		  }
		  if (j == 0) {
		    ptr[i * N] = ptr[(i-1) * N] + 3;
        continue;
		  }

      int32_t upc = ptr[(i-1) * N + j] + a_cost;
      int32_t leftc = ptr[i * N + j - 1] + b_cost;
      int32_t diagc = ptr[(i-1) * N + j - 1] + transition_cost;
      int32_t sum = std::min(upc, std::min(leftc, diagc));

      ptr[i * N + j] = sum;
    }
  }
//  std::cout << "DONE"<<std::endl;
  return ptr[0];  // TODO: FIX
}


PYBIND11_MODULE(texterrors_align,m) {
  m.doc() = "pybind11 plugin";
  m.def("calc_sum_cost", &calc_sum_cost, "Calculate summed cost matrix");
  m.def("get_best_path", &get_best_path, "get_best_path");
  m.def("lev_distance", lev_distance<int>);
  m.def("lev_distance", lev_distance<char>);
  m.def("lev_distance_str", &lev_distance_str);
}
