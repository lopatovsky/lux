#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
#include <vector>
#include <map>
#include <unordered_map>
#include <iostream>

using namespace std;

typedef pair<int,int> ii;
typedef vector<int> vi;
typedef vector<vi> vvi;
typedef vector<ii> vii;

#define REP(i,a) for (int i = 0; i < (a); i++)
#define FOR(i,a,b) for (int i = (a); i <= (b); i++)

int size = 50; // 48 plus boundaries

class Action {


};

class Factory {
public:

    void update(std::string u, int s, bool my, int pow, int x, int y, std::unordered_map<std::string, int> c){
       unit_id = u;
       strain_id = s;
       is_my = my;
       power = pow;
       px = x;
       py = y;
       cargo = c;
    }

    std::string unit_id;
    int strain_id;
    bool is_my;
    int power;
    int px;
    int py;
    // 'ice', 'ore', 'water', 'metal'
    std::unordered_map<std::string, int> cargo;
};

class Unit {
public:
    void update(std::string u, bool h, bool my, int pow, int x, int y, std::unordered_map<std::string, int> c){
       unit_id = u;
       heavy = h;
       is_my = my;
       power = pow;
       px = x;
       py = y;
       cargo = c;
    }

    std::string unit_id;
    bool is_my;
    bool heavy;  // false is light unit
    int power;
    int px;
    int py;
    std::unordered_map<std::string, int> cargo;

};

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
    CLux(py::array_t<int> _ice, py::array_t<int> _ore, int factories_per_team_):factories_per_team(factories_per_team_){
        ice = numpy_to_vector(_ice, -1);
        ore = numpy_to_vector(_ore, -1);
    }

    // A bit more effectively just take in the diff. but maybe who cares.
    void update_rubble(py::array_t<int> _rubble){
        rubble = numpy_to_vector(_rubble, 1e9);
    }

    void update_lichen(py::array_t<int> _lichen, py::array_t<int> _lichen_strains){
        lichen = numpy_to_vector(_lichen, -1);
        lichen_strains = numpy_to_vector(_lichen_strains, -1);
    }

    void update_factory(std::string u, int s, bool my, int pow, int x, int y,
                        std::unordered_map<std::string, int> c){
        auto it = factories.find(u);
        if (it == factories.end()){
            const auto& pair = factories.emplace(u, Factory());
            it = pair.first;
        }
        it->second.update(u,s,my,pow,x,y,c);
    }

    void update_unit(std::string u, bool h, bool my, int pow, int x, int y,
                     std::unordered_map<std::string, int> c){
        auto it = units.find(u);
        if (it == units.end()){
            const auto& pair = units.emplace(u, Unit());
            it = pair.first;
        }
        it->second.update(u,h,my,pow,x,y,c);
    }

    void remove_zombie_factory(std::string key){
        factories.erase(key);
    }

    void remove_zombie_unit(std::string key){
        units.erase(key);
    }

    void update_assorted(int real_step_, int step_){
        real_step = real_step_;
        step = step_;
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

    std::unordered_map <std::string, Factory> factories;
    std::unordered_map <std::string, Unit> units;

    int real_step, step, factories_per_team;
    vvi ice, ore, rubble, lichen, lichen_strains;
};


PYBIND11_MODULE(clux, m) {

    py::class_<CLux>(m, "CLux")
        .def(py::init<py::array_t<int>, py::array_t<int>, int>())
        .def("update_rubble", &CLux::update_rubble)
        .def("update_lichen", &CLux::update_lichen)
        .def("update_factory", &CLux::update_factory)
        .def("update_unit", &CLux::update_unit)
        .def("remove_zombie_factory", &CLux::remove_zombie_factory)
        .def("remove_zombie_unit", &CLux::remove_zombie_unit)
        .def("update_assorted", &CLux::update_assorted)

        .def("shortest_path", &CLux::shortest_path);
}