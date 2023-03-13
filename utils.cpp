#ifndef UTILS_CPP
#define UTILS_CPP

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

#endif