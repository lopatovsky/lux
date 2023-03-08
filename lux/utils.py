
def code_to_direction(code):
    if code == 0:
        return (0, 0)
    elif code == 1:
        return (0, -1)
    elif code == 2:
        return (1, 0)
    elif code == 3:
        return (0, 1)
    return (-1, 0)

# For inline iteration.
# [(1,(0,-1)), (2,(1,0)), (3,(0,1)), (4,(-1,0))]

def valid(i, j):
    return i >= 0 and j >= 0 and i < 48 and j < 48

def distance(x,y):
    return abs(x[0] - y[0]) + abs(x[1] - y[1])

def next_move(unit):
    if len(unit.action_queue) == 0:
        return 0  # Don't move or unknown

    next_action = unit.action_queue[0]
    px = unit.pos[0]
    py = unit.pos[1]

    if next_action[0] == 0:  # move
        return next_action[1]
    return 0

def my_turn_to_place_factory(place_first: bool, step: int):
    if place_first:
        if step % 2 == 1:
            return True
    else:
        if step % 2 == 0:
            return True
    return False


# direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
def direction_to(src, target):
    ds = target - src
    dx = ds[0]
    dy = ds[1]
    if dx == 0 and dy == 0:
        return 0
    if abs(dx) > abs(dy):
        if dx > 0:
            return 2
        else:
            return 4
    else:
        if dy > 0:
            return 3
        else:
            return 1
