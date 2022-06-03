#= Neural Network that always outputs 2 =#

include("./model.jl")
using .Model

include("./snake.jl")
using .Snake

import Random

function main()

    Random.seed!(round(Int64, time() * 1000))

    sizes::Array{UInt32} = [length(Snake.DIRS)*3, 16, 16, 4]
    GENERATIONS = 2^32

    Model.init(Snake.play, sizes, UInt32(100), 0.15, 0.1, 0.2, true, UInt32(1))

    net::Model.Network = Model.Network()

    for generation in 1:GENERATIONS
        Model.train_generation()

        Base.run(`clear`)
        println("Finished generation $(generation)/$(GENERATIONS)")
        println("DATA:")
        for pair in pairs(Model.get_generation_data())
            println(pair)
        end
        print("\n\n")

        net = Model.get_best()
        Model.write_to_file(net, "/tmp/snake")
    end

    println("Press any key to see the model in action")
    readline()
    println("Creating the plot")

    net = Model.get_best()
    Model.write_to_file(net, "/tmp/snake")
    feed(a) = Model.feedforward(net, a)
    Snake.play(feed, true)
end

main()

