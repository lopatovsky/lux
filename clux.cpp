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

bool is_in_factory( ii factory, ii pos){
    return max(abs(factory.xx - pos.xx) , abs(factory.yy - pos.yy)) <= 1;
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
       pos = {x,y};
       cargo = c;
    }

    std::string unit_id;
    int strain_id;
    bool is_my;
    int power;
    int px;
    int py;
    ii pos;
    // 'ice', 'ore', 'water', 'metal'
    std::unordered_map<std::string, int> cargo;
    vector<pair<int,ii>> rubble_vec;
    vector<pair<ii,ii>> lichen_vec;
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

    py::array_t<int> to_raw_action() const{
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

Action dig_action(int n){
    return Action(3, 0, 0, 0, 0, n);
}

Action recharge_action(int till_capacity){
    return Action(5, 0, 0, till_capacity, 0, 1);
}

Action pick_up_action(int resource_code, int pick_up_amount){
    return Action(2, 0, resource_code, pick_up_amount, 0, 1);
}

Action transfer_action(int resource_code){
    int transfer_dir = 0;
    return Action(1, transfer_dir, resource_code, 3000, 0, 1);
}


class Unit {
public:
    void update(std::string u, bool h, bool my, int pow, int x, int y,
                std::unordered_map<std::string, int>& c, std::vector<py::array_t<int>>& aq, string& ms){
       unit_id = u;
       is_heavy = h;
       is_my = my;
       power = pow;
       px = x;
       py = y;
       pos = {x,y};
       cargo = c;
       mother_ship_id = ms;
    }

    std::string unit_id;
    bool is_my;
    bool is_heavy;  // false is light unit
    int power;
    int px;
    int py;
    ii pos;
    std::unordered_map<std::string, int> cargo;
    std::vector<Action> action_queue;
    std::string mother_ship_id;
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

std::vector<py::array_t<int>> get_raw_actions(const vector<Action> & actions){
        std::vector<py::array_t<int>> raw_actions(actions.size());
        REP(i,actions.size()) raw_actions[i] = actions[i].to_raw_action();
        return raw_actions;
    }

int closest_factory(int i, int j, const std::unordered_map <std::string, Factory>& factories){
    int closest_factory = 100;

    for(const auto& f: factories){
         closest_factory = min(closest_factory, distance(f.ss.px, f.ss.py, i, j));
    }
    return closest_factory;
}

Factory * get_closest_factory(int i, int j, std::unordered_map <std::string, Factory*>& factories){
    int min_dist = 100;
    Factory * factory;
    for(auto& f: factories){
        int dist = distance(f.ss->px, f.ss->py, i, j);
        if (dist < min_dist){
            min_dist = dist;
            factory = f.ss;
        }
    }
    return factory;
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
                     std::unordered_map<std::string, int>& c, std::vector<py::array_t<int>>& aq, string& ms ){
        auto it = units.find(u);
        if (it == units.end()){
            const auto& pair = units.emplace(u, Unit());
            it = pair.first;
        }
        it->second.update(u,h,my,pow,x,y,c,aq,ms);
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
        for(auto& [key, value] : factories) {
            if (value.is_my) my_factories.emplace( key, &value);
            else his_factories.emplace( key, &value);
        }

        my_units.clear();
        his_units.clear();
        for(auto& [key, value] : units) {
            if (value.is_my) my_units[key] = &value;
            else his_units[key] = &value;
        }

        // invalidate move to opponents factories
        for(auto &it: his_factories){
            for(const auto& [x, y]: iterate_mask({it.ss->px, it.ss->py}, factory_mask)){
                if(valid(x, y)) rubble[x][y] = 1e6;
            }
        }

        divide_and_conquer_advanced();
    }


    void divide_and_conquer_advanced(){
        assigned_rubbles = board(-1e6);

        auto economy_zone = board(-1);
        auto distance_map = board(1e6);
        queue<ii> q;
        queue<int> q_dist;

        Factory* factory_strain[20];  // factory by strain_id;

        for(auto& f : factories){
            factory_strain[f.ss.strain_id] = &f.ss;
            f.ss.rubble_vec.clear();
            f.ss.lichen_vec.clear();
            for(const auto& [x, y]: iterate_mask(f.ss.pos, factory_mask)){
                q.push({x,y});
                q_dist.push(0);
                economy_zone[x][y] = f.ss.strain_id;
                economy_zone[x][y] = f.ss.strain_id;
            }
        }

        REP(i,N)REP(j,N)
            if (lichen_strains[i][j] >= 0) {
                q.push({i, j});
                q_dist.push(0);
                economy_zone[i][j] = lichen_strains[i][j];
            }

        while (!q.empty()) {
            const auto& [x,y] = q.front();
            int dist = q_dist.front();
            q.pop(); q_dist.pop();
            int ez = economy_zone[x][y];

            FOR(code,1,4){
                int nx = x + code_to_direction[code].xx;
                int ny = y + code_to_direction[code].yy;
                if (valid(nx,ny) && economy_zone[nx][ny] == -1){
                    economy_zone[nx][ny] = ez;
                    distance_map[nx][ny] = dist;
                    q.push({nx,ny});
                    q_dist.push(dist+1);
                }
            }
        }

        REP(i,N)REP(j,N)
            if(rubble[i][j] && economy_zone[i][j] >= 0){
                factory_strain[economy_zone[i][j]]->rubble_vec.PB(MP(distance_map[i][j],MP(i,j)));
            }

        // FOR LICHEN MINING:

        REP(i,N) REP(j,N)
            if(lichen_strains[i][j] >= 0 && !factory_strain[lichen_strains[i][j]]->is_my){
                Factory * lichen_fac = factory_strain[lichen_strains[i][j]];
                int dist = distance( i,j, lichen_fac->px, lichen_fac->py);

                for(auto &factory: my_factories){
                    int my_dist = distance( i,j, factory.ss->px, factory.ss->py);
                    factory.ss->lichen_vec.PB(MP(MP(dist, my_dist), MP(i, j)));
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
    std::pair<int, vector<Action>> shortest_path(int t, int x0, int y0, int x1, int y1, bool is_heavy){
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

        // subtract energy gains if it is day.
        int daily_gain = is_heavy ? 10:1;
        int t0 = t;
        REP(i,path.size()){
            energy -= is_day(t0++) ? daily_gain: 0;
        }

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

        return {energy, actions};
    }

    // Optimize on energy and turns
    std::pair<int, int> shortest_path_to_dig(int t, int x0, int y0, vvi board_map, bool is_heavy){
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

            if (board_map[x][y] > 0)
                return {x,y};

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
    }


    ii find_rubble(int px, int py){
        return MP(0,0);
    }

    vector<Action> prepare_unit(const Unit& unit, Factory* mother){
        vector<Action> actions;
        if (unit.cargo.at("ore") > 0){
            actions.PB(transfer_action(1));
        }
        if (unit.cargo.at("ice") > 0){
            actions.PB(transfer_action(0));
        }

        int max_to_take = unit.is_heavy ? 500 : 120;
        if (mother->power > 800){
            max_to_take = unit.is_heavy ? 600 : 150;
        }
        if (mother->power > 2000){
            max_to_take = unit.is_heavy ? 1400 : 150;
        }
        if (mother->power > 3500){
            max_to_take = unit.is_heavy ? 3000 : 150;
        }

        int power_to_take = 0;
        if (unit.is_heavy && unit.power < 2500){
            power_to_take = max( 0, max_to_take - unit.power);
        }else if(!unit.is_heavy && unit.power < 140){
            power_to_take = max( 0, max_to_take - unit.power);
        }
        if (power_to_take > (unit.is_heavy ? 100 : 20)){
            power_to_take = min( power_to_take, mother->power);
            actions.PB(pick_up_action(4, power_to_take));
        }
        return actions;
    }

    std::vector<py::array_t<int>> mine_resource_action(string key_id, int resource_id){
        auto& unit = units[key_id];
        auto* mother = get_closest_factory( unit.px, unit.py, my_factories);

        //# PREPARE ACTION
        if (is_in_factory(mother->pos, unit.pos)){
            auto prepare_actions = prepare_unit(unit, mother);
            if (!prepare_actions.empty()){
                return get_raw_actions(prepare_actions);
            }
        }

        auto [path_power, path_actions] = shortest_path( step, unit.px, unit.py, mother->px, mother->py, unit.is_heavy);

        //# GO HOME ACTION
        int energy_treshold = unit.is_heavy? 240 : 20;  // energy of four digs.
        if( unit.power < path_power + energy_treshold){
            return get_raw_actions(path_actions);
        }

        // Only this one line is different, lol:
        ii resource_loc = shortest_path_to_dig(step, unit.px, unit.py, (resource_id? ore: ice), unit.is_heavy);

        auto [resource_power, resource_path_actions] =
                     shortest_path( step, unit.px, unit.py, resource_loc.xx, resource_loc.yy, unit.is_heavy);

        auto [worst_case_home_power, home_actions] =
                     shortest_path( 30 /*worst_case*/, resource_loc.xx, resource_loc.yy, mother->px, mother->py, unit.is_heavy);
        int rest_power = unit.power - (path_power + worst_case_home_power);

        int dig_number = rest_power / (unit.is_heavy ? 50 : 5); // 60 -> 50 because unit gets some energy from sun.

        //# GO HOME ACTION
        if(dig_number < 1){
            return get_raw_actions(path_actions);
        }

        // Also this is different.
        int cargo_load = unit.cargo.at("ice") + unit.cargo.at("ore") + unit.cargo.at("water") + unit.cargo.at("metal");
        int cargo_space = (unit.is_heavy? 3000 : 100) - cargo_load;

        int max_digs = unit.is_heavy ? cargo_space / 20:
                                       cargo_space / 2;
        // end of different


        dig_number = min(dig_number, max_digs);

        //# GO DIG ACTION
        if (dig_number){
            resource_path_actions.PB(dig_action(dig_number));
        }

        return get_raw_actions(resource_path_actions);
    }

    std::vector<py::array_t<int>> mine_ore_action(string key_id){
        return mine_resource_action(key_id, 1);
    }

    std::vector<py::array_t<int>> mine_ice_action(string key_id){
        return mine_resource_action(key_id, 0);
    }

    ii assign_rubble( Factory * factory, int px, int py){
        int best_x = 0, best_y = 0, best_score = 1e9;

        for (auto& [dist, loc]: factory->rubble_vec){
            int last_access = assigned_rubbles[loc.xx][loc.yy];
            if (step - last_access < 50) continue;

            int rubble_value = rubble[loc.xx][loc.yy];
            int dist_from_unit = distance(px, py, loc.xx, loc.yy);

            int D = 4;
            int K = 1;
            int L = 2;

            int score = D * dist + K * rubble_value + L * dist_from_unit;

            if (score < best_score){
                best_score = score;
                best_x = loc.xx;
                best_y = loc.yy;
            }
        }
        // cerr << "For: " << px << ", " << py << "Assign rubble: " << best_x << ", " << best_y << endl;

        assigned_rubbles[best_x][best_y] = step;
        return {best_x, best_y};
    }

    ii assign_lichen( Factory * factory, int px, int py, bool is_inner){
        int best_x = 0, best_y = 0, best_score = 1e9;

        // Outer lichen eater params. Bigger means more important.
        int DH = 0;   // Distance to his factory
        int DM = 4;  // Distance to my factory
        int L = 1;   // Lichen value
        int U = 2;   // Distance to unit

        // Inner lichen eater params.
        if(is_inner){
            DH = 100;
            DM = 1;
            L = 0;
            U = 1;
        }

        for (auto& [distances, loc]: factory->lichen_vec){
            int last_access = assigned_rubbles[loc.xx][loc.yy];
            if (step - last_access < 50) continue;

            auto& [distance_to_him, distance_to_me] = distances; // Distances to factories.

            int lichen_value = lichen[loc.xx][loc.yy];
            int dist_from_unit = distance(px, py, loc.xx, loc.yy);

            int score = DH * distance_to_him +
                        DM * distance_to_me +
                        L *  lichen_value +
                        U * dist_from_unit;

            if (score < best_score){
                best_score = score;
                best_x = loc.xx;
                best_y = loc.yy;
            }
        }
        // cerr << "For: " << px << ", " << py << "Assign rubble: " << best_x << ", " << best_y << endl;

        assigned_rubbles[best_x][best_y] = step;
        return {best_x, best_y};
    }

    std::vector<py::array_t<int>> remove_rubble_action(const string& key_id){

        auto& unit = units[key_id];
        auto* mother = my_factories[unit.mother_ship_id]; //get_closest_factory( unit.px, unit.py, my_factories);

        //# PREPARE ACTION
        if (is_in_factory(mother->pos, unit.pos)){
            auto prepare_actions = prepare_unit(unit, mother);
            if (!prepare_actions.empty()){
                return get_raw_actions(prepare_actions);
            }
        }

        auto [path_power, path_actions] = shortest_path( step, unit.px, unit.py, mother->px, mother->py, unit.is_heavy);

        //# GO HOME ACTION
        int energy_treshold = unit.is_heavy? 240 : 20;  // energy of four digs.
        if( unit.power < path_power + energy_treshold){
            return get_raw_actions(path_actions);
        }

        //ii rubble_loc = shortest_path_to_dig(step, unit.px, unit.py, rubble, unit.is_heavy);
        const auto [rx, ry] = assign_rubble(mother, unit.px, unit.py);


        auto [rubble_power, rubble_path_actions] =
                     shortest_path( step, unit.px, unit.py, rx, ry, unit.is_heavy);

        auto [worst_case_home_power, home_actions] =
                     shortest_path( 30 /*worst_case*/, rx, ry, mother->px, mother->py, unit.is_heavy);
        int rest_power = unit.power - (path_power + worst_case_home_power);

        int dig_number = rest_power / (unit.is_heavy ? 50 : 5); // 60 -> 50 because unit gets some energy from sun.

        //# GO HOME ACTION
        if(dig_number < 1){
            return get_raw_actions(path_actions);
        }

        int max_digs = unit.is_heavy ? (rubble[rx][ry]+19)/ 20:
                                       (rubble[rx][ry]+1 )/  2;
        dig_number = min(dig_number, max_digs);

        //# GO DIG ACTION
        if (dig_number){
            rubble_path_actions.PB(dig_action(dig_number));
        }

        return get_raw_actions(rubble_path_actions);
    }

    std::vector<py::array_t<int>> remove_lichen_action(const string& key_id, bool is_inner){

        auto& unit = units[key_id];
        auto* mother = my_factories[unit.mother_ship_id]; //get_closest_factory( unit.px, unit.py, my_factories);

        // LICHEN specific
        if (mother->lichen_vec.empty()){
            return remove_rubble_action(key_id);
        }

        //# PREPARE ACTION
        if (is_in_factory(mother->pos, unit.pos)){
            auto prepare_actions = prepare_unit(unit, mother);
            if (!prepare_actions.empty()){
                return get_raw_actions(prepare_actions);
            }
        }

        auto [path_power, path_actions] = shortest_path( step, unit.px, unit.py, mother->px, mother->py, unit.is_heavy);

        //# GO HOME ACTION
        int energy_treshold = unit.is_heavy? 240 : 20;  // energy of four digs.
        if( unit.power < path_power + energy_treshold){
            return get_raw_actions(path_actions);
        }

        // LICHEN specific
        const auto [rx, ry] = assign_lichen(mother, unit.px, unit.py, is_inner);


        auto [rubble_power, rubble_path_actions] =
                     shortest_path( step, unit.px, unit.py, rx, ry, unit.is_heavy);

        auto [worst_case_home_power, home_actions] =
                     shortest_path( 30 /*worst_case*/, rx, ry, mother->px, mother->py, unit.is_heavy);
        int rest_power = unit.power - (path_power + worst_case_home_power);

        int dig_number = rest_power / (unit.is_heavy ? 50 : 5); // 60 -> 50 because unit gets some energy from sun.

        //# GO HOME ACTION
        if(dig_number < 1){
            return get_raw_actions(path_actions);
        }

        // LICHEN specific
        int max_digs = unit.is_heavy ? 1:
                                       (rubble[rx][ry]+9 )/  10 + 2;  // +2, because it may regrow till I come there.

        dig_number = min(dig_number, max_digs);

        //# GO DIG ACTION
        if (dig_number){
            rubble_path_actions.PB(dig_action(dig_number));
        }

        return get_raw_actions(rubble_path_actions);
    }

    std::vector<py::array_t<int>> distract_oponent_action(const string& key_id){
        std::vector<py::array_t<int>> v; return v;
    }

    std::vector<py::array_t<int>> suicide_action(const string& key_id){
        std::vector<py::array_t<int>> v; return v;
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

private:
    std::unordered_map <std::string, Factory> factories;
    std::unordered_map <std::string, Factory*> my_factories, his_factories;
    std::unordered_map <std::string, Unit> units;
    std::unordered_map <std::string, Unit*> my_units, his_units;

    int real_step, step, factories_per_team;
    vvi ice, ore, rubble, lichen, lichen_strains;

    vvi ice_dist, ore_dist, rubble_around_factory, assigned_rubbles;
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

        .def("remove_rubble_action", &CLux::remove_rubble_action)
        .def("mine_ice_action", &CLux::mine_ice_action)
        .def("mine_ore_action", &CLux::mine_ore_action)
        .def("remove_lichen_action", &CLux::remove_lichen_action)
        .def("distract_oponent_action", &CLux::distract_oponent_action)
        .def("suicide_action", &CLux::suicide_action);
}