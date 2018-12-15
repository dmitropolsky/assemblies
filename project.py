#! /usr/bin/python

from numpy.random import binomial
import heapq

n = 10000
k = 100
p = 0.01
beta = 0.05
T = 30

# stimulus to neural space is k x n
stimulus_inputs = binomial(k,p,n).astype(float)

# connectome of A (recurrent) is n x n
A_connectome = binomial(1,p,(n,n)).astype(float)

winners = []
support = set()
support_size_at_t = []
new_winners_at_t = []
# for each time step
for t in xrange(T):
	# calculate inputs into each of n neurons
	inputs = [stimulus_inputs[i] for i in xrange(n)]
	for i in winners:
		for j in xrange(n):
			inputs[j] += A_connectome[i][j]
	# identify top k winners 	
	new_winners = heapq.nlargest(k, range(len(inputs)), inputs.__getitem__)
	for i in new_winners:
		stimulus_inputs[i] *= (1+beta)
	# plasticity: for winners, for previous winners, update edge weight
	for i in winners:
		for j in new_winners:
			A_connectome[i][j] *= (1+beta)
	# update winners
	for i in new_winners:
		support.add(i)
	winners = new_winners
	support_size_at_t.append(len(support))
	if t >= 1:
		new_winners_at_t.append(support_size_at_t[-1]-support_size_at_t[-2])
