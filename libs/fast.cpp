#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <queue>


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
	int32_t* ptr = (int32_t*) buf.ptr;
	int32_t numr = array.shape()[0], numc = array.shape()[1];
	int32_t maxlen = numr + numc;
	// Find maximum value of reward matrix on last column or row. Use that point as start point of path
	// (and later append with insert steps)
	int maxreward = -maxlen;
	Pair bestpoint(numr - 1, numc - 1);
	for(int i = 0; i < numr; i++) {
		int val = ptr[i*numc + numc - 1];
		if (val > maxreward) {
//			std::cout << i << " " << val << std::endl;
			maxreward = val;
			bestpoint = Pair(i, numc-1);
		}
	}
	for(int j = 0; j < numc; j++) {
		int val = ptr[(numr-1)*numc + j];
		if (val > maxreward) {
//			std::cout << j << " " << val << std::endl;
			maxreward = val;
			bestpoint = Pair(numr-1, j);
		}
	}

	if (numr > 32000 || numc > 32000) throw std::runtime_error("Input array too large!");
	std::cout << bestpoint.i << " " << bestpoint.j << std::endl;
	int16_t i = bestpoint.i, j = bestpoint.j;
	std::queue< std::vector<Pair>> paths_to_explore;
	std::vector<Pair> bestpath;
	std::vector<Pair> path;
	int32_t best_continuous_match_len = -1;
	path.reserve(maxlen);
	path.push_back(Pair(i, j));
//	std::cout << i << " " << j << std::endl;
	int32_t max_paths_explore = 30000;
	int32_t paths_found = 0;
	while (true) {
		if (i == 0 && j == 0) {
			int32_t path_len = path.size();
			int32_t startidx = -1, endidx = -1;
			for(int32_t n = 1; n < path_len; n++) {
				Pair& pair = path[n];
				if (pair.i + 1 == path[n-1].i && pair.j + 1 == path[n-1].j) {
					if (startidx == -1) startidx = n;
					endidx = n;
				}
			}
			int32_t continuous_match_len = endidx - startidx;
			std::cout << continuous_match_len <<  std::endl;
			if (bestpath.size() == 0 || continuous_match_len < best_continuous_match_len) {
				best_continuous_match_len = continuous_match_len;
				bestpath = path;
			}
			if (paths_to_explore.size() == 0) {
				break;
			}
			path = paths_to_explore.front();
			Pair& p = path.back();
			i = p.i, j = p.j;
			paths_to_explore.pop();
		}
		int32_t upc, leftc, diagc;
		int8_t idx = -1;
		if (i == 0) {
		  idx = 1;
		} else if (j == 0) {
		  idx = 0;
		} else {
			upc = ptr[(i-1)*numc + j];
			leftc = ptr[i*numc + j - 1];
			diagc = ptr[(i-1)*numc + j - 1];
		}
		if (idx != -1) {
			;
		} else if (diagc >= leftc && diagc >= upc) {
			idx = 2;
		} else if (upc > leftc && upc != diagc && (texta[i] != textb[j] || upc - 2 > diagc)) {
			idx = 0;
		} else if (leftc > upc && leftc != diagc && (texta[i] != textb[j] || leftc - 2 > diagc)) {
		  idx = 1;
		} else {
			if (leftc == diagc && upc == diagc) {
				idx = 2;
			}
			if (leftc == diagc || upc == diagc) {
				throw std::runtime_error("Should not be possible A");
			} else if (leftc == upc) {
			  if (paths_found < max_paths_explore) {
					std::vector<Pair> pathcopied(path);
					Pair explorep(i, j - 1);
					pathcopied.push_back(explorep);
					paths_to_explore.push(pathcopied);

					pathcopied = path;
					explorep = Pair(i - 1, j);
					pathcopied.push_back(explorep);
					paths_to_explore.push(pathcopied);
					paths_found++;
				}

				idx = 2;
			} else if (leftc - 2 == diagc) {
				if (paths_found < max_paths_explore) {
					std::vector<Pair> pathcopied(path);
					Pair explorep(i, j - 1);
					pathcopied.push_back(explorep);
					paths_to_explore.push(pathcopied);
				}
				idx = 2;
			} else if (upc - 2 == diagc) {
				if (paths_found < max_paths_explore) {
					std::vector<Pair> pathcopied(path);
					Pair explorep(i - 1, j);
					pathcopied.push_back(explorep);
					paths_to_explore.push(pathcopied);
				}
				idx = 2;
			} else if (diagc <= upc && diagc <= leftc) {
				idx = 2;
			} else {
				throw std::runtime_error("Should not be possible C " + std::to_string(leftc) + " " + std::to_string(upc) + " " + std::to_string(diagc));
			}
		}

		if (idx == 0) {
			i--;
		} else if (idx == 1) {
			j--;
		} else {
			i--, j--;
		}
		Pair newp = Pair(i, j);
		path.push_back(newp);
	}

	if (bestpoint.i == numr - 1 && bestpoint.j != numc - 1) {
		std::vector<Pair> toappend;
		for(int j = bestpoint.j + 1; j < numc; j++) {
			toappend.push_back(Pair(bestpoint.i, j));
		}
		bestpath.insert(bestpath.begin(), toappend.begin(), toappend.end());
	} else if (bestpoint.j == numc - 1 && bestpoint.i != numr - 1) {
		std::vector<Pair> toappend;
		for(int i = bestpoint.i + 1; i < numr; i++) {
			toappend.push_back(Pair(i, bestpoint.j));
		}
		bestpath.insert(bestpath.begin(), toappend.begin(), toappend.end());  // if the path were in right order append
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
			int32_t elem_cost = -1;
		  if (texta[i] == textb[j]) elem_cost = 1;

			if (i == 0) {
				if (j == 0) {
					ptr[0] = 0;
				} else {
					ptr[j] = ptr[j - 1] + elem_cost;
				}
			} else if (j == 0) {
				ptr[i * N] = ptr[(i - 1) * N] + elem_cost;
			} else if (i == 1 && j == 1) {
				if (elem_cost == -1) {
					elem_cost = -2;
				}
				ptr[i * N + j] = elem_cost;
			} else {
				int32_t upc = ptr[(i-1) * N + j];
		    int32_t leftc = ptr[i * N + j - 1];
			  int32_t diagc = ptr[(i-1) * N + j - 1];
			  int32_t transition_cost = std::max(upc, std::max(leftc, diagc));
				if (diagc > leftc && diagc > upc) {
					if (elem_cost == -1) elem_cost = -2;
					transition_cost += elem_cost;
				} else {
					transition_cost += -1;
				}
		    ptr[i * N + j] = transition_cost;
			}
    }
  }
  return py::cast<py::none>(Py_None);
}


PYBIND11_MODULE(fast,m) {
  m.doc() = "pybind11 plugin";
  m.def("calc_sum_cost", &calc_sum_cost, "Calculate summed cost matrix");
  m.def("get_best_path", &get_best_path, "get_best_path");
}
