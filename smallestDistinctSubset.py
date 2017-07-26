"""
n = {1 ... 10}
a = {1, 2, 3}
b = {3, 4, 5}

sol -> {2, 4}
sol cant be {2, 3} b/c that is a subset of one of the m subsets
sol could also be {1, 4}, {1, 5},


a = {1, 2, 3}
b = {3, 4, 5}
c = {5, 6, 7}


if the m subsets don't cover n, then the simplest solution would always be a set containing one of the missing elemenets.
2^n possible subsets
Iterate through all subsets check if they are a subset of any of the b subsets.

"""
from random import sample
from itertools import combinations

a = sample(xrange(1000), 200)
b = sample(a, 20)
set(b).issubset(a)

def findsubsets(S, m):
	return set(combinations(S, m))
