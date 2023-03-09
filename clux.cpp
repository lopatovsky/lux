#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
#include <vector>

using namespace std;

typedef pair<int,int> ii;
typedef vector<int> vi;
typedef vector<vi> vvi;
typedef vector<ii> vii;

#define REP(i,a) for (int i = 0; i < (a); i++)
#define FOR(i,a,b) for (int i = (a); i <= (b); i++)

int size = 50; // 48 plus boundaries

vvi numpy_to_vector(py::array_t<int> board, int border_value){
    if (board.ndim() != 2 || board.shape(0) != 48 || board.shape(1) != 48) {
        throw std::runtime_error("Input matrix must have shape (48, 48)");
    }
   // Convert the input matrix to a 2D vector
    vvi matrix(size, vi(size, border_value));
    auto ptr = static_cast<int *>(board.request().ptr);
    for (int i = 0; i < board.shape(0); ++i) {
        for (int j = 0; j < board.shape(1); ++j) {
            matrix[i+1][j+1] = ptr[i * board.shape(1) + j];
        }
    }
    return matrix;
}


class CLux {
public:
    CLux(py::array_t<int> _ice, py::array_t<int> _ore){
        ice = numpy_to_vector(_ice, 0);
        ore = numpy_to_vector(_ore, 0);
    }

    void update_rubble(py::array_t<int> _rubble){
        rubble = numpy_to_vector(_rubble, 1e9);
    }

    int shortest_path(int x1, int y1, int x2, int y2, bool is_heavy){
        return shortest_path_(x1+1,y1+1,x2+1,y2+1,is_heavy);
    }

    // TODO factories

private:
    int shortest_path_(int x1, int y1, int x2, int y2, bool is_heavy){
       vvi distances(size, vi(size, 0));

       REP(j,size){
           REP(k,size){
                int i = j;
                if (ice[i][j] != 1){
                    distances[j][k] += ice[i][j] + ore[j][k] + rubble[i][k];
                }
       }};
       int out = 0;
       REP(i,size) REP(j,size) out+= distances[i][j];
       return out;
    }

    vvi ice, ore, rubble;
};

PYBIND11_MODULE(clux, m) {
    py::class_<CLux>(m, "CLux")
        .def(py::init<py::array_t<int>, py::array_t<int>>())
        .def("update_rubble", &CLux::update_rubble)
        .def("shortest_path", &CLux::shortest_path);
}