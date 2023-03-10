#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
#include <vector>
#include <queue>
#include <map>
#include <unordered_map>
#include <iostream>
#include <tuple>
#include <cstdlib>
#include <climits>

using namespace std;

typedef pair<int,int> ii;
typedef vector<int> vi;
typedef vector<vi> vvi;
typedef vector<ii> vii;

#define MP make_pair
#define PB push_back
#define ff first
#define ss second

#define xx first
#define yy second

#define REP(i,a) for (int i = 0; i < (a); i++)
#define FOR(i,a,b) for (int i = (a); i <= (b); i++)

int N = 48;

bool valid(int i, int j){
    return i >= 0 && j >= 0 && i < N && j < N;
}

int distance(int x0, int y0, int x1, int y1){
    return abs(x0 - x1) + abs(y0 - y1);
}

std::tuple<int, int> code_to_direction(int code) {
    switch(code) {
        case 0:
            return std::make_tuple(0, 0);
        case 1:
            return std::make_tuple(0, -1);
        case 2:
            return std::make_tuple(1, 0);
        case 3:
            return std::make_tuple(0, 1);
        default:
            return std::make_tuple(-1, 0);
    }
}

vvi board(){
    vvi b(N, vi(N));
    return b;
}

vvi board(int val){
    vvi b(N, vi(N, val));
    return b;
}

    // LOL
    const vvi& factory_surrounding_mask = {
        {0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0},
        {0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0},
        {0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0},
        {0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0},
        {1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1},
        {0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0},
        {0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0},
        {0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0},
        {0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0},
    };


    const vvi& factory_close_mask = {
        {0, 1, 1, 1, 0},
        {1, 0, 0, 0, 1},
        {1, 0, 0, 0, 1},
        {1, 0, 0, 0, 1},
        {0, 1, 1, 1, 0}
    };

    const vvi& factory_mask = {
        {1, 1, 1},
        {1, 1, 1},
        {1, 1, 1}
    };

    vii iterate_mask( ii center, const vvi& mask){
        int mask_size = mask.size();
        int shift = mask_size / 2;
        vii points;
        REP(i,mask_size)
            REP(j,mask_size){
                if (mask[i][j]){
                    points.PB(MP(center.xx + i - shift,
                                 center.yy + j - shift));
                }
            }
        return points;
    }

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

class Action {
public:
    Action(py::array_t<int> raw_action){
        type = *(raw_action.data(0));
        dir_code = *(raw_action.data(1));
        std::tie(dx, dy) = code_to_direction(dir_code);
        resource = *(raw_action.data(2));
        amount = *(raw_action.data(3));
        repeat = *(raw_action.data(4));
        n = *(raw_action.data(5));
    }

    int type, dir_code, dx, dy, resource, amount, repeat, n;
};

class Unit {
public:
    void update(std::string u, bool h, bool my, int pow, int x, int y,
                std::unordered_map<std::string, int> c, std::vector<py::array_t<int>> aq){
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
    std::vector<Action> action_queue;
};

vvi numpy_to_vector(py::array_t<int> array, int border_value){
    if (array.ndim() != 2 || array.shape(0) != N || array.shape(1) != N) {
        throw std::runtime_error("Input matrix must have shape (48, 48)");
    }
   // Convert the input matrix to a 2D vector
    vvi matrix = board();
    auto ptr = static_cast<int *>(array.request().ptr);
    for (int i = 0; i < array.shape(0); ++i) {
        for (int j = 0; j < array.shape(1); ++j) {
            matrix[i][j] = ptr[i * array.shape(1) + j];
        }
    }
    return matrix;
}

vvi distance_map(vvi map){
    auto distance_map = board(INT_MAX);

    queue<ii> q;
    queue<int> q_dist;
    REP(i,N)REP(j,N)
        if (map[i][j] > 0) {
            q.push({i, j});
            q_dist.push(0);
        }

    while (!q.empty()) {
        int x = q.front().xx;
        int y = q.front().yy;
        int dist = q_dist.front();
        q.pop(); q_dist.pop();

        if (!valid(x,y) || distance_map[x][y] < INT_MAX) continue;
        distance_map[x][y] = dist;

        for(auto& n: {MP(x-1, y), MP(x+1, y), MP(x, y-1), MP(x, y+1)}){
          q.push(n);
          q_dist.push(dist+1);
        }
    }

    return distance_map;
}

void print_board(vvi board){
    REP(i,N){
        REP(j,N){
            cerr << board[j][i] << "|";
        }
        cerr << endl;
    }
}

vvi get_rubble_around_factory_map(vvi rubble){
    auto acc_map = board(0);
    REP(i,N)REP(j,N){
        int rubble_sum = 0;
        for(const auto& [x, y]: iterate_mask({i,j}, factory_surrounding_mask)){
           if (valid(x,y)){
               rubble_sum += rubble[x][y];
           }else{
               rubble_sum += 100;  // Out of boundaries consider to be heavy rubble.
           }

        }
        acc_map[i][j] = rubble_sum;
    }
    print_board(acc_map);
    return acc_map;
}


class CLux {
public:
    CLux(py::array_t<int> _ice, py::array_t<int> _ore, int factories_per_team_):factories_per_team(factories_per_team_){
        ice = numpy_to_vector(_ice, -1);
        ore = numpy_to_vector(_ore, -1);

        ice_dist = distance_map(ice);
        ore_dist = distance_map(ore);
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
                     std::unordered_map<std::string, int> c, std::vector<py::array_t<int>> aq){
        auto it = units.find(u);
        if (it == units.end()){
            const auto& pair = units.emplace(u, Unit());
            it = pair.first;
        }
        it->second.update(u,h,my,pow,x,y,c,aq);
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

    int factory_placement_value(int i, int j){
        if(rubble_around_factory.empty()){
            rubble_around_factory = get_rubble_around_factory_map(rubble);
        }

        for(const auto& [x, y]: iterate_mask({i,j},factory_mask) ){
            if( !valid(x,y) || ice[x][y] || ore[x][y]) return 1000000;
        }

        int closest_factory = 100;

        for(const auto& f: factories){
             // cerr << "c++ " << i << " " << j << " " << distance(f.ss.px, f.ss.py, i, j) << " - " << factories.size() << endl;
             closest_factory = min(closest_factory, distance(f.ss.px, f.ss.py, i, j));
        }
        if (closest_factory <= 7) return 1000000;

        int ice_distance = 100;
        int ore_distance = 100;

        for(const auto& [x, y]: iterate_mask({i,j}, factory_close_mask)){
            if (valid(x,y)){
                ice_distance = min(ice_distance, ice_dist[x][y]);
                ore_distance = min(ore_distance, ore_dist[x][y]);
            }
        }

        int rubble_acc = rubble_around_factory[i][j];

        return ice_distance * 2000 + ore_distance + rubble_acc;
    }

    ii place_factory(){
        int min_val = INT_MAX;
        int x = 0,y = 0;
        REP(i,N)REP(j,N){
            int place_value = factory_placement_value(i,j);
            if (place_value < min_val){
                min_val = place_value;
                x=i; y=j;
            }
        }
        return {x,y};
    }


    // TODO factories

private:
    int shortest_path_(int x1, int y1, int x2, int y2, bool is_heavy){
       int out=0;
       return out;
    }

    std::unordered_map <std::string, Factory> factories;
    std::unordered_map <std::string, Unit> units;

    int real_step, step, factories_per_team;
    vvi ice, ore, rubble, lichen, lichen_strains;

    vvi ice_dist, ore_dist, rubble_around_factory;
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

        .def("place_factory", &CLux::place_factory)
        .def("shortest_path", &CLux::shortest_path);
}