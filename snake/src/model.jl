module Model

    include("./network.jl")

    using Base.Threads
    import Base.Threads.@threads

    using Statistics

    mutable struct MODEL
        NET_SIZES
        AGENTS_PER_GEN
        TOP_PERCENT
        MUTATION_CHANCE
        MUTATION_LIMIT
        TRAIN_FUNCTION
        MULTITHREAD
        CURRENT_GENERATION
        CURRENT_BEST
        GENERATION_DATA
        MULTIPARAM
    end

    MODEL() = MODEL(nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing)

    const STATE = Ref{MODEL}()

    function init(
        train_function::Function,
        net_sizes::Array{UInt32},
        agents_per_gen::UInt32 = UInt32(1000),
        top_percent::Float64 = 0.15,
        mutation_chance::Float64 = 0.05,
        mutation_limit::Float64 = 0.1,
        multithread::Bool = true,
        multiparam::UInt32 = UInt32(1)
    )
        STATE[] = MODEL(
            net_sizes,
            agents_per_gen,
            top_percent,
            mutation_chance,
            mutation_limit,
            train_function,
            multithread,
            nothing,
            nothing,
            nothing,
            multiparam,
        )
    end

    function get_next_generation(scores)
        cross(net1, net2) = crossover(net1, net2, STATE[].MUTATION_CHANCE, STATE[].MUTATION_LIMIT)
        getg(i) = STATE[].CURRENT_GENERATION[i]

        n = length(scores)
        top_count = floor(UInt32, n*STATE[].TOP_PERCENT)


        scores_only = [x[2] for x in scores]
        probabs = cumsum(scores_only ./ sum(scores_only))
        r() = scores[findfirst(probabs .> rand())][1]
        f(i, j) = cross(getg(i), getg(j))
        g() = rand() < 0.5 ? f(r(), r()) : cross(getg(scores[rand(1:top_count)][1]), getg(scores[rand(1:top_count)][1]))

        return [i <= top_count ? getg(scores[i][1]) : g() for i in 1:n]
    end

    function train_generation()

        if STATE[].CURRENT_GENERATION === nothing
            STATE[].CURRENT_GENERATION = [Network(STATE[].NET_SIZES) for _ in 1:STATE[].AGENTS_PER_GEN]
        end

        n = length(STATE[].CURRENT_GENERATION)
        scores = [(i, -Inf) for i in 1:n]

        if STATE[].MULTITHREAD
            @threads for i in 1:n
                feed(input) = feedforward(STATE[].CURRENT_GENERATION[i], input)
                scores[i] = (i, mean([STATE[].TRAIN_FUNCTION(feed) for _ in 1:STATE[].MULTIPARAM]))
            end
        else
            for i in 1:n
                feed(input) = feedforward(STATE[].CURRENT_GENERATION[i], input)
                scores[i] = (i, STATE[].TRAIN_FUNCTION(feed))
            end
        end

        sort!(scores, by = x->x[2], rev=true)

        STATE[].CURRENT_BEST = STATE[].CURRENT_GENERATION[scores[1][1]]
        STATE[].GENERATION_DATA = Dict("max" => scores[1][2], "avg" => mean([x[2] for x in scores]), "median" => median([x[2] for x in scores]) )

        STATE[].CURRENT_GENERATION = get_next_generation(scores)
    end

    function get_best()
        return STATE[].CURRENT_BEST
    end

    function get_generation_data()
        return STATE[].GENERATION_DATA
    end

end