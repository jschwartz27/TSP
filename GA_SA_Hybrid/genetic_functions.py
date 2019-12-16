import random
from math import exp
from copy import deepcopy


def flatten(l):
    return [item for sublist in l for item in sublist]


def evolve(N, chrom_n, gen_n, data, optimal):
    # chroms are each a list of the point index
    # which will be used to get coords from data
    chroms = [list(range(N)) for i in range(5000)]
    for i in chroms:
        random.shuffle(i)
    genome = list(map(lambda x: eval_distance(x, data), chroms))
    genome.sort()

    best_Fitness, der_Übermensch = genome[0][0], genome[0][1]
    genome = genome[:chrom_n]
    print("Gen_0:: {}".format(best_Fitness))

    for gen in range(gen_n):
        T = Temperatures(gen, N)
        selected_pop, elite = random_selection(genome, T)
        selected_pop_noF = list(zip(*selected_pop))[1]
        next_gen = crossover(selected_pop_noF, T, data)
        genome = elite + next_gen
        genome.sort()
        if genome[0][0] < best_Fitness:
            der_Übermensch = genome[0][1]
            best_Fitness = round(genome[0][0], 2)

        print("Gen{}:: {}, T:: {}      \r".format(gen+1, best_Fitness, T), end="")
    print("Final_Fitness:: {}".format(best_Fitness))
    error = round(((best_Fitness - optimal)/optimal) * 100, 2)
    print("Error:: {}%\n".format(error))

    return der_Übermensch


def Temperature(n, l):
    p = round(.25 * exp(-n/30), 2)
    return max(int(p * 100), 1)

### SELECTION ###


def random_selection(DNA, T):
    # TODO evaluation done here for all
    l = len(DNA)
    # Elitism
    if random.random() < .4:
        # remember n_elite % 2 MUST == 0
        n_elite = 2
        elite = DNA[:n_elite]
        # chroms = DNA[:n_elite]
    else:
        elite = list()
        n_elite = 0
    # new pop will be recombined with elite after crossover
    l_adj = l - n_elite

    if random.random() < .3:
        # truncates by 50%
        DNA = _truncation(DNA, l)

    selection_functions = (
        _linear_rank, _exponential_rank,
        _tournament, _roulette, _boltzmann
    )
    # w = (.0, .0, .2, .8, .0)
    w = (.1, .2, .3, .4, 0)
    # w = (.05, .25, .25, .4, .05)
    func = random.choices(population=selection_functions,
                          weights=w)[0]
    # feeds in DNA with fitness tuple
    # TODO change this to keyword args
    keywords = {
        "DNA": DNA,
        "length": l,
        "length_adj": l_adj,
        "temp": T
    }

    return func(**keywords), elite


def _truncation(DNA, l, n=.5):
    assert l % 2 == 0, "CANNOT TRUNCATE UNEAVEN DNA"

    DNA = DNA[:int(l*n)]
    # this doubles and maintains fitness ordering of chroms
    return flatten(zip(DNA, DNA))


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
        fs = [1 for i in range(len(fs))]
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

### CROSSOVER ###


def crossover(DNA, T, data):
    DNA = list(DNA)
    random.shuffle(DNA)
    half = int(len(DNA)/2)
    L = range(len(DNA[0]))
    chrom_l = len(DNA[0])

    cross_DNA = flatten(list(map(
        lambda x, y: PMX_crossover(x, y, chrom_l), DNA[:half], DNA[half:]
    )))

    genome = list(map(lambda x: eval_distance(x, data), cross_DNA))

    # +2 because it needs to be at least len 2 for a reverse
    mut_len = round(chrom_l * (T/100)) + 2

    return list(map(
        lambda x: mutation(x, chrom_l, mut_len, T, data), genome
    ))


def _pmx_function(c1, c2, idxs):
    copy_1 = deepcopy(c1)
    splice2 = c2[idxs[0]:idxs[1]]
    for i in splice2:
        copy_1.remove(i)
    return copy_1[:idxs[0]] + splice2 + copy_1[idxs[0]:]


def PMX_crossover(lover_1, lover_2, L):
    # https://www.researchgate.net/figure/Partially-mapped-crossover-operator-PMX_fig1_226665831
    # crossover probability of .95
    if random.random() < .95:
        idxs = random.sample(range(L), 2)
        idxs.sort()
        c1 = _pmx_function(lover_1, lover_2, idxs)
        c2 = _pmx_function(lover_2, lover_1, idxs)

        return c1, c2
    else:
        return lover_1, lover_2

### MUTATION ###


def mutation(chrom, L, mut_len, T, data):
    # mutation length is the percentage of chrom based on Temperature
    m = deepcopy(chrom[1])
    if random.random() < .7:
        i = random.choice(range(L - mut_len))
        idxs = (i, i + mut_len)
        splice = m[idxs[0]:idxs[1]]
        if random.random() < .5:
            random.shuffle(splice)
        else:
            splice.reverse()
        m[idxs[0]:idxs[1]] = splice
    else:
        #i_1 = random.randint(len(chrom) - 4)
        #i_2 = random.randint(len(chrom) - 9)
        j = random.randrange(len(chrom[1]))
        k = random.randrange(len(chrom[1]))
        
        m[j] = chrom[1][k]
        m[k] = chrom[1][j]

    m = eval_distance(m, data)
    # TODO Check that the negs or whatevs actually make since since we reducing
    if m[0] < chrom[0] or random.random() < T/170:#random.random() < exp((m[0]-chrom[0])/T):
        #print(T)
        #print(exp((m[0]-chrom[0])/T))

        return m
    else:
        return chrom


def _swap_mutation(chrom, L, mut_len, T):
    pass


### FITNESS ###


def eval_distance(chrom, data):
    # This is to add the return to the start point
    c = chrom + [chrom[0]]
    distance = 0
    for i in range(1, len(c)):
        distance += _euc_2d(data[c[i]], data[c[i-1]])

    return (round(distance, 2), chrom)


def _euc_2d(p1, p2):
    return round(((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**.5, 2)


def Temperatures(n, l):
    eigth = (l/4)/100
    p = round(eigth * exp(((-n/30) / (l/30))), 2)

    return max(int(p * 100), 1) +1

if __name__ == '__main__':
    g = 1000
    l = list(map(lambda x: Temperatures(x, 131), range(g)))
    print(l)
