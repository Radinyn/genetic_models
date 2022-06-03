using JLD2
using Distributions

struct Network
    num_of_layers
    sizes
    weights
    biases
end

function _create_network(sizes::Array{UInt32})
    n = length(sizes)

    d = Uniform(-1.0, 1.0)
    r(y, x) = rand(d, y, x)

    biases = [zeros(y, 1) for y in sizes[2:n]]
    weights = [r(sizes[i+1], sizes[i]) ./ sqrt(sizes[i]) for i in 1:(n-1)]

    return Network(
        n,
        sizes,
        weights,
        biases
    )
end

Network(sizes::Array{UInt32}) = _create_network(sizes)
Network() = Network(nothing, nothing, nothing, nothing)

sigmoid(x) = 1.0/(1.0+exp(-x))
relu(x) = max(0, x)

activation = relu

function feedforward(net::Network, a)
    for (b, w) in zip(net.biases, net.weights)
        a = activation.(w * a) .+ b
    end

    return a
end


function crossover(net1::Network, net2::Network, mutation_chance::Float64, strength_limit::Float64)
    net3 = Network(net1.sizes)

    for i in 1:length(net1.biases)
        for j in 1:length(net1.biases[i])
            if rand() < 0.5
                net3.biases[i][j] = net1.biases[i][j]
            else
                net3.biases[i][j] = net2.biases[i][j]
            end
        end
    end

    for i in 1:length(net1.weights)
        for j in 1:length(net1.weights[i])
            if rand() < 0.5
                net3.weights[i][j] = net1.weights[i][j]
            else
                net3.weights[i][j] = net2.weights[i][j]
            end
        end
    end

    mutate!(net3, mutation_chance, strength_limit)
    return net3
end

function mutate!(net::Network, chance::Float64, strength_limit::Float64)

    d = Uniform(-strength_limit, strength_limit)

    filter(a, b) = b ? a : 0.0
    clamp_neuron(x) = x

    for i in 1:length(net.weights)
        s = size(net.weights[i])
        mutation = rand(d, s...)
        mask = rand(s...) .< chance
        mutation = filter.(mutation, mask)
        net.weights[i] = clamp_neuron.(net.weights[i] .+ mutation)
    end

    for i in 1:length(net.biases)
        s = size(net.biases[i])
        mutation = rand(d, s...)
        mask = rand(s...) .< chance
        mutation = filter.(mutation, mask)
        net.biases[i] = clamp_neuron.(net.biases[i] .+ mutation)
    end
end

function write_to_file(net::Network, filepath::String)
    @save filepath net
end

function read_from_file(filepath::String)
    @load filepath net
    return net
end