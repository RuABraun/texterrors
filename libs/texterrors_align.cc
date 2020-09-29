#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <queue>
#include <iostream>


namespace py = pybind11;


struct Pair {
		Pair() {}
    Pair(int16_t f_, int16_t s_) {
        i = f_;
        j = s_;
    }
    int16_t i;
    int16_t j;
};


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
    Pair newp = Pair(i, j);
    path.push_back(newp);
	}

	if (bestpath.size() == 1) throw std::runtime_error("No best path found!");
	for(int32_t k = 0; k < bestpath.size(); k++) {
		bestpath_lst.append(bestpath[k].i);
		bestpath_lst.append(bestpath[k].j);
	}
}


py::object calc_sum_cost(py::array_t<int32_t> array, std::vector<int32_t>& texta, std::vector<int32_t>& textb) {
	if ( array.ndim() != 2 )
    throw std::runtime_error("Input should be 2-D NumPy array");

  int M = array.shape()[0], N = array.shape()[1];
  if (M != texta.size() || N != textb.size()) throw std::runtime_error("Sizes do not match!");
  auto buf = array.request();
  int32_t* ptr = (int32_t*) buf.ptr;

  for(int32_t i = 0; i < M; i++) {
		for(int32_t j = 0; j < N; j++) {
			int32_t transition_cost = 1;
		  if (texta[i] == textb[j]) transition_cost = 0;

		  if (i == 0 && j == 0) {
		    ptr[0] = 0;
        continue;
		  }
		  if (i == 0)  {
		    ptr[j] = ptr[j - 1] + transition_cost;
        continue;
		  }
		  if (j == 0) {
		    ptr[i * N] = ptr[(i-1) * N] + transition_cost;
        continue;
		  }

      int32_t upc = ptr[(i-1) * N + j] + transition_cost;
      int32_t leftc = ptr[i * N + j - 1] + transition_cost;
      int32_t diagc = ptr[(i-1) * N + j - 1] + 2 * transition_cost;
      int32_t sum = std::min(upc, std::min(leftc, diagc));

      ptr[i * N + j] = sum;
    }
  }
  return py::cast<py::none>(Py_None);
}


PYBIND11_MODULE(texterrors_align,m) {
  m.doc() = "pybind11 plugin";
  m.def("calc_sum_cost", &calc_sum_cost, "Calculate summed cost matrix");
  m.def("get_best_path", &get_best_path, "get_best_path");
}
