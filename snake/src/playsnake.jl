#= Neural Network that always outputs 2 =#

include("./model.jl")
using .Model

include("./snake.jl")
using .Snake

function main()
    net = Model.read_from_file("../jld/snake.jld2")
    feed(a) = Model.feedforward(net, a)
    Snake.play(feed, true)
end

main()

