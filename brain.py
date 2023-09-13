#! /usr/bin/python3.9
# DOWNLOADED FROM https://github.com/dmitropolsky/assemblies
import numpy as np
import heapq
from collections import defaultdict
from scipy.stats import binom
from scipy.stats import truncnorm
from scipy.stats import norm
import math
import random

# Configurable assembly model for simulations
# Author Daniel Mitropolsky, 2018

class Stimulus:
	def __init__(self, k):
		self.k = k

class Area:
	def __init__(self, name, n, k, beta=0.05):
		self.name = name
		self.n = n
		self.k = k
		# Default beta
		self.beta = beta
		# Betas from stimuli into this area.
		self.stimulus_beta = {}
		# Betas from areas into this area.
		self.area_beta = {}
		self.w = 0
		# List of winners currently (after previous action). Can be 
		# read by caller.
		self.winners = []
		self.new_w = 0
		# new winners computed DURING a projection, do not use outside of internal project function
		self.new_winners = []
		# list of lists of all winners in each round
		self.saved_winners = []
		# list of size of support in each round
		self.saved_w = []
		self.num_first_winners = -1
		# Whether to fix (freeze) the assembly (winners) in this area
		self.fixed_assembly = False
		# Whether to fully simulate this area
		self.explicit = False

	def update_winners(self):
		self.winners = self.new_winners
		if not self.explicit:
			self.w = self.new_w

	def update_stimulus_beta(self, name, new_beta):
		self.stimulus_beta[name] = new_beta

	def update_area_beta(self, name, new_beta):
		self.area_beta[name] = new_beta

	def fix_assembly(self):
		if not self.winners:
			raise ValueError('Area %s does not have assembly; cannot fix.' % self.name)
			return
		self.fixed_assembly = True

	def unfix_assembly(self):
		self.fixed_assembly = False

class Brain:
	def __init__(self, p, save_size=True, save_winners=False):
		self.areas = {}
		self.stimuli = {}
		self.stimuli_connectomes = {}
		self.connectomes = {} 
		self.p = p
		self.save_size = save_size
		self.save_winners = save_winners
		# For debugging purposes in applications (eg. language)
		self.no_plasticity = False
		self.no_print=False
		self.use_normal_ppf=False

	def add_stimulus(self, name, k):
		self.stimuli[name] = Stimulus(k)
		new_connectomes = {}
		for key in self.areas:
			if self.areas[key].explicit:
				new_connectomes[key] = np.random.binomial(k, self.p, size=(self.areas[key].n)) * 1.0
			else:
				new_connectomes[key] = np.empty(0)
			self.areas[key].stimulus_beta[name] = self.areas[key].beta
		self.stimuli_connectomes[name] = new_connectomes

	def add_area(self, name, n, k, beta):
		self.areas[name] = Area(name, n, k, beta)

		for stim_name, stim_connectomes in self.stimuli_connectomes.items():
			stim_connectomes[name] = np.empty(0)
			self.areas[name].stimulus_beta[stim_name] = beta

		new_connectomes = {}
		for key in self.areas:
			other_area_size = 0
			if self.areas[key].explicit:
				other_area_size = self.areas[key].n
			new_connectomes[key] = np.empty((0, other_area_size))
			if key != name:
				self.connectomes[key][name] = np.empty((other_area_size, 0))
			# by default use beta for plasticity of synapses from this area to other areas
			# by default use other area's beta for synapses from other area to this area
			self.areas[key].area_beta[name] = self.areas[key].beta
			self.areas[name].area_beta[key] = beta
		self.connectomes[name] = new_connectomes

	def add_explicit_area(self, name, n, k, beta, custom_inner_p=None, custom_out_p=None, custom_in_p=None):
		self.areas[name] = Area(name, n, k, beta)
		self.areas[name].explicit = True

		for stim_name, stim_connectomes in self.stimuli_connectomes.items():
			stim_connectomes[name] = np.random.binomial(self.stimuli[stim_name].k, self.p, size=(n)) * 1.0
			self.areas[name].stimulus_beta[stim_name] = beta

		if custom_inner_p:
			inner_p = custom_inner_p
		else:
			inner_p = self.p

		if custom_in_p:
			in_p = custom_in_p
		else:
			in_p = self.p

		if custom_out_p:
			out_p = custom_out_p
		else:
			out_p = self.p 

		new_connectomes = {}
		for key in self.areas:
			if key == name:  # create explicitly
				new_connectomes[key] = np.random.binomial(1, inner_p, size=(n,n)) * 1.0
			if key != name:  
				if self.areas[key].explicit:
					other_n = self.areas[key].n
					new_connectomes[key] = np.random.binomial(1, out_p, size=(n, other_n)) * 1.0
					self.connectomes[key][name] = np.random.binomial(1, in_p, size=(other_n, n)) * 1.0
				else: # we will fill these in on the fly
					# TODO: if explicit area added late, this will not work
					# But out_p to a non-explicit area must be default p, for fast sampling to work
					new_connectomes[key] = np.empty((n,0))
					self.connectomes[key][name] = np.empty((0,n))
			self.areas[key].area_beta[name] = self.areas[key].beta
			self.areas[name].area_beta[key] = beta
		self.connectomes[name] = new_connectomes
		# Explicitly set w to n so that all computations involving this area are explicit.
		self.areas[name].w = n

	def update_plasticity(self, from_area, to_area, new_beta):
		self.areas[to_area].area_beta[from_area] = new_beta

	def update_plasticities(self, area_update_map={}, stim_update_map={}):
		# area_update_map consists of area1: list[ (area2, new_beta) ]
		# represents new plasticity FROM area2 INTO area1
		for to_area, update_rules in area_update_map.items():
			for (from_area, new_beta) in update_rules: 
				self.update_plasticity(from_area, to_area, new_beta)

		# stim_update_map consists of area: list[ (stim, new_beta) ]f
		# represents new plasticity FROM stim INTO area
		for area, update_rules in stim_update_map.items():
			for (stim, new_beta) in update_rules:
				self.areas[area].stimulus_beta[stim] = new_beta

	def activate(self, area_name, index):
		area = self.areas[area_name]
		k = area.k 
		assembly_start = k*index
		area.winners = list(range(assembly_start, assembly_start+k))
		area.fix_assembly()

	def project(self, stim_to_area, area_to_area, verbose=False):
		# Validate stim_area, area_area well defined
		# stim_to_area: {"stim1":["A"], "stim2":["C","A"]}
		# area_to_area: {"A":["A","B"],"C":["C","A"]}

		stim_in = defaultdict(lambda: [])
		area_in = defaultdict(lambda: [])

		for stim, areas in stim_to_area.items():
			if stim not in self.stimuli:
				raise IndexError(stim + " not in brain.stimuli")
				return
			for area in areas:
				if area not in self.areas:
					raise IndexError(area + " not in brain.areas")
					return
				stim_in[area].append(stim)
		for from_area, to_areas in area_to_area.items():
			if from_area not in self.areas:
				raise IndexError(from_area + " not in brain.areas")
				return
			for to_area in to_areas:
				if to_area not in self.areas:
					raise IndexError(to_area + " not in brain.areas")
					return
				area_in[to_area].append(from_area)

		to_update = set().union(stim_in.keys(), area_in.keys())

		for area in to_update:
			num_first_winners = self.project_into(self.areas[area], stim_in[area], area_in[area], verbose)
			self.areas[area].num_first_winners = num_first_winners
			if self.save_winners:
				self.areas[area].saved_winners.append(self.areas[area].new_winners)

		# once done everything, for each area in to_update: area.update_winners()
		for area in to_update:
			self.areas[area].update_winners()
			if self.save_size:
				self.areas[area].saved_w.append(self.areas[area].w)

	def project_into(self, area, from_stimuli, from_areas, verbose=False):
	# projecting everything in from stim_in[area] and area_in[area]
	# calculate: inputs to self.connectomes[area] (previous winners)
	# calculate: potential new winners, Binomial(sum of in sizes, k-top)
	# k top of previous winners and potential new winners
	# if new winners > 0, redo connectome and intra_connectomes 
	# have to wait to replace new_winners
		if not self.no_print:
			print("Projecting " + ",".join(from_stimuli) + " and " + ",".join(from_areas) + " into " + area.name)

		# If projecting from area with no assembly, complain.
		for from_area in from_areas:
			if not self.areas[from_area].winners or (self.areas[from_area].w == 0):
				raise Exception("Projecting from area with no assembly: " + from_area)

		name = area.name
		prev_winner_inputs = [0.] * area.w
		for stim in from_stimuli:
			stim_inputs = self.stimuli_connectomes[stim][name]
			for i in range(area.w):
				prev_winner_inputs[i] += stim_inputs[i]
		for from_area in from_areas:
			connectome = self.connectomes[from_area][name]
			for w in self.areas[from_area].winners:
				for i in range(area.w):
					prev_winner_inputs[i] += connectome[w][i]

		if verbose:
			print("prev_winner_inputs: ")
			print(prev_winner_inputs)

		# simulate area.k potential new winners if the area is not explicit 
		if not area.explicit:
			total_k = 0
			input_sizes = []
			num_inputs = 0
			normal_approx_mean = 0.0 
			normal_approx_var = 0.0
			for stim in from_stimuli:
				local_k = self.stimuli[stim].k
				total_k += local_k
				input_sizes.append(local_k)
				num_inputs += 1
				if self.use_normal_ppf:
					local_p = self.custom_stim_p[stim][name]
					normal_approx_mean += local_k * local_p
					normal_approx_var += ((local_k * local_p * (1-p)) ** 2)
			for from_area in from_areas:
				#if self.areas[from_area].w < self.areas[from_area].k:
				#	raise ValueError("Area " + from_area + "does not have enough support.")
				effective_k = len(self.areas[from_area].winners)
				total_k += effective_k
				input_sizes.append(effective_k)
				num_inputs += 1
				if self.use_normal_ppf:
					local_p = self.custom_stim_p[from_area][name]
					normal_approx_mean += effective_k * local_p
					normal_approx_var += ((effective_k * local_p * (1-p)) ** 2)

			if verbose:
				print("total_k = " + str(total_k) + " and input_sizes = " + str(input_sizes))

			effective_n = area.n - area.w
			# Threshold for inputs that are above (n-k)/n percentile.
			percentage = (float(effective_n-area.k)/effective_n)
			if self.use_normal_ppf:
				# each normal approximation is N(n*p, n*p*(1-p))
				normal_approx_std = math.sqrt(normal_approx_var)
				alpha = binom.ppf(percentage, loc=normal_approx_mean, scale=normal_approx_std)
			else:
				# self.p can be changed to have a custom connectivity into this brain area
				# but all incoming areas' p must be the same
				alpha = binom.ppf(percentage, total_k, self.p)
			if verbose:
				print("Alpha = " + str(alpha))
			# use normal approximation, between alpha and total_k, round to integer
			# create k potential_new_winners
			if self.use_normal_ppf:
				mu = normal_approx_mean
				std = normal_approx_std
			else:
				mu = total_k * self.p
				std = math.sqrt(total_k * self.p * (1.0-self.p))
			a = float(alpha - mu) / std
			b = float(total_k - mu) / std
			potential_new_winners = truncnorm.rvs(a, b, scale=std, size=area.k)
			for i in range(area.k):
				potential_new_winners[i] += mu
				potential_new_winners[i] = round(potential_new_winners[i])
			potential_new_winners = potential_new_winners.tolist()

			if verbose:
				print("potential_new_winners: ")
				print(potential_new_winners)

			# take max among prev_winner_inputs, potential_new_winners
			# get num_first_winners (think something small)
			# can generate area.new_winners, note the new indices
			all_potential_winners = prev_winner_inputs + potential_new_winners
		else:
			all_potential_winners = prev_winner_inputs

		new_winner_indices = heapq.nlargest(area.k, range(len(all_potential_winners)), all_potential_winners.__getitem__)
		num_first_winners = 0

		if not area.explicit:
			first_winner_inputs = []
			for i in range(area.k):
				if new_winner_indices[i] >= area.w:
					first_winner_inputs.append(potential_new_winners[new_winner_indices[i] - area.w])
					new_winner_indices[i] = area.w + num_first_winners
					num_first_winners += 1
		area.new_winners = new_winner_indices
		area.new_w = area.w + num_first_winners

		# For experiments with a "fixed" assembly in some area.
		if area.fixed_assembly:
			area.new_winners = area.winners
			area.new_w = area.w
			first_winner_inputs = []
			num_first_winners = 0

		# print name + " num_first_winners = " + str(num_first_winners)

		if verbose:
			print("new_winners: ")
			print(area.new_winners)

		# for i in num_first_winners
		# generate where input came from
			# 1) can sample input from array of size total_k, use ranges
			# 2) can use stars/stripes method: if m total inputs, sample (m-1) out of total_k
		first_winner_to_inputs = {}
		for i in range(num_first_winners):
			input_indices = random.sample(range(0, total_k), int(first_winner_inputs[i]))
			inputs = np.zeros(num_inputs)
			total_so_far = 0
			for j in range(num_inputs):
				inputs[j] = sum([((total_so_far + input_sizes[j]) > w >= total_so_far) for w in input_indices])
				total_so_far += input_sizes[j]
			first_winner_to_inputs[i] = inputs
			if verbose:
				print("for first_winner # " + str(i) + " with input " + str(first_winner_inputs[i]) + " split as so: ")
				print(inputs)

		m = 0
		# connectome for each stim->area
			# add num_first_winners cells, sampled input * (1+beta)
			# for i in repeat_winners, stimulus_inputs[i] *= (1+beta)
		for stim in from_stimuli:
			if num_first_winners > 0:
				self.stimuli_connectomes[stim][name] = np.resize(self.stimuli_connectomes[stim][name],
					area.w + num_first_winners)
			for i in range(num_first_winners):
				self.stimuli_connectomes[stim][name][area.w + i] = first_winner_to_inputs[i][m]
			stim_to_area_beta = area.stimulus_beta[stim]
			if self.no_plasticity:
				stim_to_area_beta = 0.0
			for i in area.new_winners:
				self.stimuli_connectomes[stim][name][i] *= (1+stim_to_area_beta)
			if verbose:
				print(stim + " now looks like: ")
				print(self.stimuli_connectomes[stim][name])
			m += 1

		# !!!!!!!!!!!!!!!!
		# BIG TO DO: Need to update connectomes for stim that are NOT in from_stimuli
		# For example, if last round fired areas A->B, and stim has never been fired into B.

		# connectome for each in_area->area
			# add num_first_winners columns
			# for each i in num_first_winners, fill in (1+beta) for chosen neurons
			# for each i in repeat_winners, for j in in_area.winners, connectome[j][i] *= (1+beta)
		for from_area in from_areas:
			from_area_w = self.areas[from_area].w
			from_area_winners = self.areas[from_area].winners
			self.connectomes[from_area][name] = np.pad(self.connectomes[from_area][name], 
				((0,0),(0,num_first_winners)), 'constant', constant_values=0)
			for i in range(num_first_winners):
				total_in = first_winner_to_inputs[i][m]
				sample_indices = random.sample(from_area_winners, int(total_in))
				for j in range(from_area_w):
					if j in sample_indices:
						self.connectomes[from_area][name][j][area.w+i] = 1.0
					if j not in from_area_winners:
						self.connectomes[from_area][name][j][area.w+i] = np.random.binomial(1, self.p)
			area_to_area_beta = area.area_beta[from_area]
			if self.no_plasticity:
				area_to_area_beta = 0.0
			for i in area.new_winners:
				for j in from_area_winners:
					self.connectomes[from_area][name][j][i] *= (1.0 + area_to_area_beta)
			if verbose:
				print("Connectome of " + from_area + " to " + name + " is now:")
				print(self.connectomes[from_area][name])
			m += 1

		# expand connectomes from other areas that did not fire into area
		# also expand connectome for area->other_area
		for other_area in self.areas:
			if other_area not in from_areas:
				self.connectomes[other_area][name] = np.pad(self.connectomes[other_area][name], 
					((0,0),(0,num_first_winners)), 'constant', constant_values=0)
				for j in range(self.areas[other_area].w):
					for i in range(area.w, area.new_w):
						self.connectomes[other_area][name][j][i] = np.random.binomial(1, self.p)
			# add num_first_winners rows, all bernoulli with probability p
			self.connectomes[name][other_area] = np.pad(self.connectomes[name][other_area],
				((0, num_first_winners),(0, 0)), 'constant', constant_values=0)
			columns = (self.connectomes[name][other_area]).shape[1]
			for i in range(area.w, area.new_w):
				for j in range(columns):
					self.connectomes[name][other_area][i][j] = np.random.binomial(1, self.p)
			if verbose:
				print("Connectome of " + name + " to " + other_area + " is now:")
				print(self.connectomes[name][other_area])

		return num_first_winners










	

