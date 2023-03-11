#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
#include <vector>
#include <queue>
#include <map>
#include <algorithm>
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
typedef vector<vii> vvii;

#define MP make_pair
#define PB push_back
#define ff first
#define ss second

#define xx first
#define yy second

#define REP(i,a) for (int i = 0; i < (a); i++)
#define FOR(i,a,b) for (int i = (a); i <= (b); i++)

int N = 48;

bool is_day(int step){
    int mod = step % 50;
    return mod < 30;
}

bool valid(int i, int j){
    return i >= 0 && j >= 0 && i < N && j < N;
}

int distance(int x0, int y0, int x1, int y1){
    return abs(x0 - x1) + abs(y0 - y1);
}

vii code_to_direction = {{0, 0}, {0, -1}, {1, 0}, {0, 1}, {-1, 0}};

vvi board(){
    vvi b(N, vi(N));
    return b;
}

vvi board(int val){
    vvi b(N, vi(N, val));
    return b;
}

int step_price(int rubble_value, bool heavy){
    // light cost of moving: floor( 1 + 0.05*rubble )
    // heavy cost of moving: floor( 20 + 1*rubble )
    return heavy ? 20 + rubble_value : 1 + rubble_value/20;
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
        std::tie(dx, dy) = code_to_direction[dir_code];
        resource = *(raw_action.data(2));
        amount = *(raw_action.data(3));
        repeat = *(raw_action.data(4));
        n = *(raw_action.data(5));
    }

    Action(int type_, int dir_code_, int resource_, int amount_, int repeat_, int n_): type(type_), dir_code(dir_code_),
                                resource(resource_), amount(amount_), repeat(repeat_), n(n_){}

    py::array_t<int> to_raw_action(){
        py::array_t<int> result({6});
        int* data = result.mutable_data();
        data[0] = type;
        data[1] = dir_code;
        data[2] = resource;
        data[3] = amount;
        data[4] = repeat;
        data[5] = n;
        return result;
    }

    int type, dir_code, dx, dy, resource, amount, repeat, n;
};

Action move_action(int dir_code, int n, int repeat = 0){
    return Action(0, dir_code, 0, 0, repeat, n);
}

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

void print_board(const vvi& board){
    REP(i,N){
        REP(j,N){
            cerr << (board[j][i] == 1e6 ? 0 : board[j][i]) << "|";
        }
        cerr << endl;
    }
}

void print_board(const vector<vector<float>>& double_vec){
    vvi int_vec;

    for (const auto& row : double_vec) {
        vector<int> int_row;
        for (const auto& elem : row) {
            int_row.push_back(static_cast<int>(elem));
        }
        int_vec.push_back(int_row);
    }
    print_board(int_vec);
}

int closest_factory(int i, int j, const std::unordered_map <std::string, Factory>& factories){
    int closest_factory = 100;

    for(const auto& f: factories){
         closest_factory = min(closest_factory, distance(f.ss.px, f.ss.py, i, j));
    }
    return closest_factory;
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

        my_factories.clear();
        his_factories.clear();
        for(auto it: factories) {
            if (it.ss.is_my) my_factories[it.ff] = &it.ss;
            else his_factories[it.ff] = &it.ss;
        }

        my_units.clear();
        his_units.clear();
        for(auto it: units) {
            if (it.ss.is_my) my_units[it.ff] = &it.ss;
            else his_units[it.ff] = &it.ss;
        }

        // invalidate move to opponents factories
        for(auto it: his_factories){
            for(const auto& [x, y]: iterate_mask({it.ss->px, it.ss->py}, factory_mask)){
                if(valid(x, y)) rubble[x][y] = 1e6;
            }
        }
    }

    // optimize on shortest path and energy

    // TODO this path may still be ignoring his. factories
    std::vector<py::array_t<int>> shortest_path_2(int x0, int y0, int x1, int y1, bool is_heavy){
        // int step = TODO for energy

        // TODO for hypercorrect ..should be triple< energy, len, turns. maybe shorter path exists with more turns.

        vvii distance_map(N, vii(N, MP(1e6,1e6)));  // map of (shortest_path, energy) pairs
        auto visited = board(0);
        auto dir = board(0);

        priority_queue<pair<ii,ii>, vector<pair<ii,ii>>, greater<pair<ii,ii>> > q;
        q.push(MP(MP(0,0),MP(x0, y0)));  // (shortest_path, energy) (pos_x, pos_y)
        dir[x0][y0] = 0;  // holds direction from which it was visited 1-4.
        distance_map[x0][y0] = MP(0,0);

        while (!q.empty()) {

            const auto& [my_obj, point] = q.top();
            int my_dist = my_obj.xx, my_energy = my_obj.yy;
            int x = point.xx, y = point.yy;
            q.pop();

            if (x==x1 && y==y1) break;

            if(visited[x][y]) continue;
            visited[x][y] = 1;
            // int my_dir = dir[x][y];

            FOR(code,1,4){
                 int nx = x + code_to_direction[code].xx;
                 int ny = y + code_to_direction[code].yy;
                 if (valid(nx,ny) && !visited[nx][ny]){
                    int dist = my_dist + 1;
                    int price = step_price(rubble[nx][ny], is_heavy);  // TODO!!! NO GO factories.
                    int energy = my_energy + price;

                    ii obj = MP(dist, energy);

                    if (obj < distance_map[nx][ny]){
                        distance_map[nx][ny] = obj;
                        dir[nx][ny] = code;
                        q.push(MP(obj,MP(nx, ny)));
                    }
                 }
            }
        }

        // print_board(dir);
        // print_board(distance_map);

        int energy = distance_map[x1][y1].ff;
        int turns = distance_map[x1][y1].ss;

        int x = x1;
        int y = y1;

        int len = 0;
        vector<int> path;

        while( dir[x][y] != 0){
            path.PB(dir[x][y]);
            const auto& p = code_to_direction[dir[x][y]];
            x = x - p.xx;
            y = y - p.yy;
            ++len;
        }
        reverse(path.begin(), path.end());

        vector<Action> actions;
        int start_segment = 0;
        path.PB(0);  // padding for last segment

        int path_size = path.size();
        FOR(i,1, path_size-1){
          if(path[i] != path[i-1]){
            actions.PB(move_action(path[i-1], i - start_segment));
            start_segment = i;
          }
        }

        std::vector<py::array_t<int>> raw_actions(actions.size());
        REP(i,actions.size()) raw_actions[i] = actions[i].to_raw_action();

        return raw_actions;
    }

    // Optimize on energy and turns
    std::vector<py::array_t<int>> shortest_path(int x0, int y0, int x1, int y1, bool is_heavy){
        // int step = TODO for energy

        // TODO for hypercorrect ..should be triple< energy, len, turns. maybe shorter path exists with more turns.

        vvii distance_map(N, vii(N, MP(1e6,1e6)));  // map of (distance, turn/segment) pairs
        auto visited = board(0);
        auto dir = board(0);

        priority_queue<pair<ii,ii>, vector<pair<ii,ii>>, greater<pair<ii,ii>> > q;
        q.push(MP(MP(0,0),MP(x0, y0)));  // (dist + turns/segment count) (pos_x, pos_y)
        dir[x0][y0] = 0;  // holds direction from which it was visited 1-4.
        distance_map[x0][y0] = MP(0,0);

        while (!q.empty()) {

            const auto& [my_obj, point] = q.top();
            int my_dist = my_obj.xx, my_turns = my_obj.yy;
            int x = point.xx, y = point.yy;
            q.pop();

            if (x==x1 && y==y1) break;

            if(visited[x][y]) continue;
            visited[x][y] = 1;
            int my_dir = dir[x][y];

            FOR(code,1,4){
                 int nx = x + code_to_direction[code].xx;
                 int ny = y + code_to_direction[code].yy;
                 if (valid(nx,ny) && !visited[nx][ny]){
                    int turns = (my_dir == code) ? my_turns : my_turns + 1;
                    int price = step_price(rubble[nx][ny], is_heavy);  // TODO!!! NO GO factories.
                    int dist = my_dist + price;

                    ii obj = MP(dist, turns);

                    if (obj < distance_map[nx][ny]){
                        distance_map[nx][ny] = obj;
                        dir[nx][ny] = code;
                        q.push(MP(obj,MP(nx, ny)));
                    }
                 }
            }
        }

        // print_board(dir);
        // print_board(distance_map);

        int energy = distance_map[x1][y1].ff;
        int turns = distance_map[x1][y1].ss;

        int x = x1;
        int y = y1;

        int len = 0;
        vector<int> path;

        while( dir[x][y] != 0){
            path.PB(dir[x][y]);
            const auto& p = code_to_direction[dir[x][y]];
            x = x - p.xx;
            y = y - p.yy;
            ++len;
        }
        reverse(path.begin(), path.end());

        vector<Action> actions;
        int start_segment = 0;
        path.PB(0);  // padding for last segment

        int path_size = path.size();
        FOR(i,1, path_size-1){
          if(path[i] != path[i-1]){
            actions.PB(move_action(path[i-1], i - start_segment));
            start_segment = i;
          }
        }

        std::vector<py::array_t<int>> raw_actions(actions.size());
        REP(i,actions.size()) raw_actions[i] = actions[i].to_raw_action();

        return raw_actions;
    }

    int factory_placement_value(int i, int j){
        if(rubble_around_factory.empty()){
            rubble_around_factory = get_rubble_around_factory_map(rubble);
        }

        for(const auto& [x, y]: iterate_mask({i,j},factory_mask) ){
            if( !valid(x,y) || ice[x][y] || ore[x][y]) return 1000000;
        }

        int factory_dist = closest_factory(i,j, factories);
        if (factory_dist <= 7) return 1000000;

        int ice_distance = 100;
        int ore_distance = 100;

        for(const auto& [x, y]: iterate_mask({i,j}, factory_close_mask)){
            if (valid(x,y)){
                ice_distance = min(ice_distance, ice_dist[x][y]);
                ore_distance = min(ore_distance, ore_dist[x][y]);
            }
        }

        // todo use my factory_dist in the function. seed 45

        int rubble_acc = rubble_around_factory[i][j];

        return ice_distance * 1000 + ore_distance + rubble_acc / 150;
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

private:
    std::unordered_map <std::string, Factory> factories;
    std::unordered_map <std::string, Factory*> my_factories, his_factories;
    std::unordered_map <std::string, Unit> units;
    std::unordered_map <std::string, Unit*> my_units, his_units;

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