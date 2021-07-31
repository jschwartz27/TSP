import random
import datetime
import statistics
from math import exp
from copy import deepcopy
from functions import flatten


def evolve(n, params, data, optimal):
    fit, mean = list(), list()
    # chromosomes are each a list of the point index
    # which will be used to get coords from data
    begin = datetime.datetime.now()
    genome = create_chroms(5000, n, data)
    best_fitness, der_uebermensch = genome[0][0], genome[0][1]
    genome = genome[:params["chrom_n"]]
    mean.append(statistics.mean(list(map(lambda x: x[0], genome))))
    fit.append(best_fitness)

    print("Gen_0:: {}".format(best_fitness))
    for gen in range(params["gen_n"]):
        t = temperatures(gen, params["gen_n"], n)
        selected_pop, elite = random_selection(genome, t)
        selected_pop_no_f = list(zip(*selected_pop))[1]
        next_gen = crossover(selected_pop_no_f, t, params["crossP"], data)
        genome = elite + next_gen
        genome.sort()
        if genome[0][0] < best_fitness:
            der_uebermensch = genome[0][1]
            best_fitness = round(genome[0][0], 2)

        print("Gen{}:: {}, T:: {}     \r".format(
            gen+1, best_fitness, t), end="")

        mean.append(statistics.mean(list(map(lambda x: x[0], genome))))
        fit.append(best_fitness)

        if best_fitness == 564:
            print("\n\tYOU WON! Proceed to the next level!")
            quit()

    print("Final_Fitness:: {}".format(best_fitness))
    error = round(((best_fitness - optimal)/optimal) * 100, 2)
    print("Error:: {}%\n".format(error))

    end = datetime.datetime.now()
    the_time = end-begin
    print("TIME:: {}".format(the_time))

    return der_uebermensch, {"Mean": mean, "Fitness": fit}, error


# ### SELECTION ###


def random_selection(dna, T):
    # TODO evaluation done here for all
    local_array = len(dna)
    # Elitism
    if random.random() < .2:
        # remember n_elite % 2 MUST == 0
        n_elite = random.choice((2, 4, 6))
        elite = dna[:n_elite]
    else:
        elite = list()
        n_elite = 0
    # new pop will be recombined with elite after crossover
    l_adj = local_array - n_elite

    if random.random() < .2:  # .3
        # truncates by 50%
        dna = _truncation(dna, local_array)

    selection_functions = (
        _linear_rank, _exponential_rank,
        _tournament, _roulette, _boltzmann
    )
    if T >= 3:
        w = (.1, .3, .2, .4, 0)
    else:
        w = (0, .5, .1, .4, 0)
    func = random.choices(population=selection_functions,
                          weights=w)[0]
    # feeds in DNA with fitness tuple
    # TODO change this to keyword args
    keywords = {
        "DNA": dna,
        "length": local_array,
        "length_adj": l_adj,
        "temp": T
    }

    return func(**keywords), elite


def _truncation(dna, local_array, n=.5):
    assert local_array % 2 == 0, "CANNOT TRUNCATE UNEAVEN DNA"

    dna = dna[:int(local_array*n)]
    # this doubles and maintains fitness ordering of chroms
    return flatten(zip(dna, dna))


def _tournament(**kwargs):
    # print(kwargs["length_adj"])
    new_chroms = list()
    for i in range(kwargs["length_adj"]):
        # this may or may not be a good idea
        size = random.choice(range(2, 4))
        cs = random.sample(kwargs["DNA"], k=size)
        new_chroms.append(max(cs))

    return new_chroms


def _roulette(**kwargs):
    # strip to get fitnesses
    fs = list(zip(*kwargs["DNA"]))[0]
    if len(list(set(fs))) == 1:
        fs = [1 for _ in range(len(fs))]
    else:
        diff = fs[-1]
        # last chrom has 0 prob of selection, but...
        fs = list(map(lambda x: x - diff, fs))
        # here, I bump up last fitness by a notch so not zero
        fs[-1] = fs[-2]/4
    fs = list(map(lambda x: x**2, fs))

    total = sum(fs)
    probs = list(map(lambda x: x/total, fs))

    return random.choices(kwargs["DNA"], probs, k=kwargs["length_adj"])


def _linear_rank(**kwargs):
    ns = list(range(1, kwargs["length"]+1))[::-1]
    probs = list(map(lambda x: x/sum(ns), ns))

    return random.choices(kwargs["DNA"], probs, k=kwargs["length_adj"])


def _exponential_rank(**kwargs):
    ns = list(range(1, kwargs["length"]+1))[::-1]
    e = list(map(lambda x: .5**(5-x), ns))
    probs = list(map(lambda x: x/sum(e), e))

    return random.choices(kwargs["DNA"], probs, k=kwargs["length_adj"])


def _boltzmann(**kwargs):
    new_chroms = list()
    for i in range(kwargs["length_adj"]):
        cs = random.sample(kwargs["DNA"], k=2)
        if cs[0] == max(cs):
            new_chroms.append(cs[0])
        else:
            if random.random() < exp((cs[0][0]-cs[1][0])/kwargs["temp"]):
                new_chroms.append(cs[0])
            else:
                new_chroms.append(cs[1])

    return new_chroms


# ### CROSSOVER ###


def crossover(dna, temperature, cross_prob, data):
    dna = list(dna)
    random.shuffle(dna)
    half = int(len(dna)/2)
    # L = range(len(dna[0]))
    chrom_l = len(dna[0])

    cross_dna = flatten(list(map(
        lambda x, y: pmx_crossover(x, y, chrom_l, cross_prob, data),
        dna[:half], dna[half:]
    )))

    genome = list(map(lambda x: eval_distance(x, data), cross_dna))

    # +2 because it needs to be at least len 2 for a reverse
    mut_len = round(chrom_l * (temperature/100)) + 2

    return list(map(
        lambda x: mutation(x, chrom_l, mut_len, temperature, data), genome
    ))


def pmx_crossover(lover_1, lover_2, local_array, cross_prob, data):
    # https://www.researchgate.net/figure/Partially-mapped-crossover-operator-PMX_fig1_226665831
    if random.random() < cross_prob:  # Crossover probability
        idxs = random.sample(range(local_array), 2)
        idxs.sort()
        c1 = _pmx_function(lover_1, lover_2, idxs)
        c2 = _pmx_function(lover_2, lover_1, idxs)

        if random.random() < .3:
            return c1, c2
        c1 = eval_distance(c1, data)
        c2 = eval_distance(c2, data)
        l1 = eval_distance(lover_1, data)
        l2 = eval_distance(lover_2, data)
        alles = [c1, c2, l1, l2]
        alles.sort()

        return alles[0][1], alles[1][1]
    else:
        return lover_1, lover_2


def _pmx_function(c1, c2, idxs):
    copy_1 = deepcopy(c1)
    splice2 = c2[idxs[0]:idxs[1]]
    for i in splice2:
        copy_1.remove(i)
    return copy_1[:idxs[0]] + splice2 + copy_1[idxs[0]:]


# ### MUTATION ###


def mutation(chrom, local_array, mut_len, temperature, data):
    # mutation length is the percentage of chrom based on Temperature
    m = deepcopy(chrom[1])
    if temperature > 4:  # <6
        pro_i = .4
        pro = .8  # .85
    else:
        pro_i = .5
        pro = .9  # .97

    random_p = random.random()
    if random_p < pro_i:
        i = random.choice(range(local_array - mut_len))
        idxs = (i, i + mut_len)
        splice = m[idxs[0]:idxs[1]]
        if random.random() < .3:
            random.shuffle(splice)
        else:
            splice.reverse()
        m[idxs[0]:idxs[1]] = splice
        if temperature < 7 and random.random() < .7:
            i = random.choice(range(local_array - mut_len))
            idxs = (i, i + mut_len)
            splice = m[idxs[0]:idxs[1]]
            if random.random() < .3:
                random.shuffle(splice)
            else:
                splice.reverse()
            m[idxs[0]:idxs[1]] = splice
    elif random_p < .45:
        for i in range(int(temperature/2)):
            p = random.choice(m)
            m.remove(p)
            m.insert(random.randrange(local_array-1), p)
    elif random_p < pro:
        xs = random.sample(range(local_array), 4)
        xs.sort()
        m[xs[0]:xs[1]] = m[xs[0]:xs[1]][::-1]
        m[xs[2]:xs[3]] = m[xs[2]:xs[3]][::-1]
    else:
        xs = random.sample(range(local_array), 6)
        xs.sort()
        m[xs[0]:xs[1]] = m[xs[0]:xs[1]][::-1]
        m[xs[2]:xs[3]] = m[xs[2]:xs[3]][::-1]
        m[xs[4]:xs[5]] = m[xs[4]:xs[5]][::-1]

    m = eval_distance(m, data)

    if m[0] < chrom[0] or random.random() < temperature/170:
        return m
    else:
        return chrom


'''
def _swap_mutation(chrom, local_array, mut_len, temperature):
    pass
'''

# ### FITNESS ###


def eval_distance(chrom, data):
    # This is to add the return to the start point
    c = chrom + [chrom[0]]
    distance = 0
    for i in range(1, len(c)):
        distance += _euc_2d(data[c[i]], data[c[i-1]])

    return round(distance, 2), chrom


def _euc_2d(p1, p2):
    return round(((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**.5, 2)


def temperatures(n, gens, length: int):
    gens_a = gens*.25
    eighth = (length/8)/100
    p = round(eighth * exp(((-n/gens_a) / (length/30))), 2)  # 30 instead of 80

    return max(int(p * 100), 1) + 1


def create_chroms(k, n, data):
    chroms = [list(range(n)) for _ in range(k)]
    for i in chroms:
        random.shuffle(i)
    genome = list(map(lambda x: eval_distance(x, data), chroms))
    genome.sort()

    return genome


if __name__ == '__main__':
    g = 6000
    array = list(map(lambda x: temperatures(x, g, 131), range(g)))
    print(array)
