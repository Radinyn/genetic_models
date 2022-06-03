module Snake

println("Loading Plots.jl...")
using Plots
using Colors

### CONSTANTS

## BOARD
N = 17

## TILES
UP = 0.1
DOWN = 0.2
LEFT = 0.3
RIGHT = 0.4
SNAKE = 0.5
APPLE = -1.
NONE = -0.1
WALL = 2.

## COLORS
APPLE_COLOR = RGB(1., 0., 0.)
SNAKE_COLOR = RGB(0., 1., 0.)
BACKGROUND_COLOR = RGB(0.1, 0.1, 0.1)
WALL_COLOR = RGB(0.3, 0.3, 0.3)

## OTHER
MAX_MOVES = 1000

MOVES = Dict(
    UP => (0, -1),
    DOWN => (0, 1),
    LEFT => (-1, 0),
    RIGHT => (1, 0)    
)

COLORS = Dict(
    UP => SNAKE_COLOR,
    DOWN => SNAKE_COLOR,
    LEFT => SNAKE_COLOR,
    RIGHT => SNAKE_COLOR,
    SNAKE => SNAKE_COLOR,
    NONE => BACKGROUND_COLOR,
    WALL => WALL_COLOR,
    APPLE => APPLE_COLOR,
)

## OTHER
MAX_MOVES = 1000

MOVES = Dict(
    UP => (0, -1),
    DOWN => (0, 1),
    LEFT => (-1, 0),
    RIGHT => (1, 0)    
)

INV_MOVES = Dict(
    (0, -1) => DOWN,
    (0, 1) => UP,
    (-1, 0) => RIGHT,
    (1, 0) => LEFT    
)

DIRS = [
    (1, 0), (0, 1), (-1, 0), (0, -1),
    (1, 1), (-1, -1), (-1, 1), (1, -1)
]

SCORE = 2
BONUS = 1
VALID = 0
LOSE = -1

### MAIN

# Process the board at head and tail, return (state, new head, new_tail, apple)
function process_board!(board, head, tail, apple)    
    next = head .+ MOVES[board[head...]]
    is_valid(x) = (1 <= x[1] <= N)&&(1 <= x[2] <= N)

    WIN_STATE = VALID

    if !is_valid(next)
        return (LOSE, (-1, -1), (-1, -1), (-1, -1))
    elseif board[next...] in (UP, DOWN, LEFT, RIGHT, WALL)
        return (LOSE, (-1, -1), (-1, -1), (-1, -1))
    elseif board[next...] == APPLE
        WIN_STATE = SCORE
    end

    board[next...] = board[head...]

    dist(a, b) = hypot(abs(a[1]-b[1]), abs(a[2]-b[2])) 

    i, j = apple
    new_tail = tail
    if WIN_STATE != SCORE
        new_tail = tail .+ MOVES[board[tail...]]
        board[tail...] = NONE        
        WIN_STATE = dist(head, apple) > dist(next, apple) ? BONUS : VALID

    else # plant new apple
        i, j = rand([(i, j) for i in 1:N for j in 1:N if board[i, j] == NONE])
        board[i, j] = APPLE
    end

    return (WIN_STATE, next, new_tail, (i, j))
end


function dist(board, head, type, dir)
    is_valid(x) = (1 <= x[1] <= N) && (1 <= x[2] <= N)
    condition(a) = type == SNAKE ? (0 < a < 1) : (a == type)

    d = 0.
    pos = head .+ dir
    while true
        if !is_valid(pos)
            return 1.
        elseif condition(board[pos...])
            return d/N
        else
            pos = pos .+ dir
            d += 1. 
        end
    end
    return 1.
end

function get_sight_vector(board, head)
    return [ dist(board, head, type, dir) for dir in DIRS for type in (APPLE, WALL, SNAKE)]
end



function play(feed, draw = false)
    board = [((max(i, j) == N) || (min(i, j) == 1)) ? WALL : NONE for i in 1:N, j in 1:N]

    in_bounds(x) = (1 <= x[1] <= N) && (1 <= x[2] <= N) && (board[x...] == NONE)
    dirs_in_bound(pos) = [x for x in [(pos .+ dir, dir) for dir in ((1, 0), (-1, 0), (0, 1), (0, -1))] if in_bounds(x[1])]

    head = (rand(2:N-1), rand(2:N-1))
    board[head...] = rand([UP, DOWN, LEFT, RIGHT])

    body, dir = rand(dirs_in_bound(head))
    board[body...] = INV_MOVES[dir]

    tail, dir = rand(dirs_in_bound(body))
    board[tail...] = INV_MOVES[dir]

    # Plant first apple
    i, j = rand([(i, j) for i in 1:N for j in 1:N if board[i, j] == NONE])
    board[i, j] = APPLE
    apple = (i, j)

    fitness(apples, steps) = max(0., (steps - 0.25*(steps^1.5)*(apples^1.2)) +  (2^apples+500*apples) ) 
    steps = 0
    apples = 0

    score = 0.0

    timer = 100

    while timer > 0
        output = feed( get_sight_vector(board, head) )
        move = findmax(output)[2]

        if move == CartesianIndex(1, 1)
            board[head...] =  UP
        elseif move == CartesianIndex(2, 1)
            board[head...] = DOWN
        elseif move == CartesianIndex(3, 1)
            board[head...] = LEFT
        else # move == output[4]
            board[head...] = RIGHT
        end
        
        state, head, tail, apple = process_board!(board, head, tail, apple)

        if state == LOSE
            return fitness(apples, steps)
        elseif state == SCORE
            timer += 100
            apples += 1
        end

        steps += 1

        if draw
            display(plot(0:1:N, 0:1:N, map(x->COLORS[x], board), showaxis=false))
            sleep(0.1)
        end

        timer -= 1
    end

    return fitness(apples, steps)
end

end