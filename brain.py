# TODOs:
# - [ ] We can make ._new_w and ._new_winners function-local;
#       they are only used inside .project.
# - [ ] We might want to turn .winners into a
#       numpy.ndarray(dtype=numpy.uint32) for efficiency.


import numpy as np
import heapq
import collections
from scipy.stats import binom
from scipy.stats import truncnorm
from scipy.stats import norm
import math
import types

# Configurable assembly model for simulations
# Author Daniel Mitropolsky, 2018

EMPTY_MAPPING = types.MappingProxyType({})

class Area:
  """A brain area.

  Attributes:
    name: the area's name (symbolic tag).
    n: number of neurons in the area.
    k: number of neurons that fire in this area.
    beta: Default value for activation-`beta`.
    beta_by_stimulus: Mapping from stimulus-name to corresponding beta.
      (In original code: `.stimulus_beta`).
    beta_by_stimulus: Mapping from area-name to corresponding beta.
      (In original code: `.area_beta`).
    w: Number of neurons that has ever fired in this area.
    saved_w: List of per-round size-of-support.
    winners: List of winners, as set by previous action.
    saved_winners: List of lists of all winners, per-round.
    num_first_winners: ??? TODO(tfish): Clarify.
    fixed_assembly: Whether the assembly (of winners) in this area
      is considered frozen.
    explicit: Whether to fully simulate this area (rather than performing
      a sparse-only simulation).
  """
  def __init__(self, name, n, k, *,
               beta=0.05, w=0, explicit=False):
    """Initializes the instance.

    Args:
      name: Area name (symbolic tag), must be unique.
      n: number of neurons(?)
      k: number of firing neurons when activated.
      beta: default activation-beta.
      w: initial 'winner' set-size.
      explicit: boolean indicating whether the area is 'explicit'
        (fully-simulated).
    """
    self.name = name
    self.n = n
    self.k = k
    self.beta = beta
    self.beta_by_stimulus = {}
    self.beta_by_area = {}
    self.w = w
    # Value of `w` since the last time that `.project()` was called.
    self._new_w = 0
    self.saved_w = []
    self.winners = []
    # Value of `winners` since the last time that `.project()` was called.
    # only to be used inside `.project()` method.
    self._new_winners = []
    self.saved_winners = []
    self.num_first_winners = -1
    self.fixed_assembly = False
    self.explicit = explicit

  def _update_winners(self):
    self.winners = self._new_winners
    if not self.explicit:
      self.w = self._new_w

  def update_beta_by_stimulus(self, name, new_beta):
    self.beta_by_stimulus[name] = new_beta

  def update_area_beta(self, name, new_beta):
    self.beta_by_area[name] = new_beta

  def fix_assembly(self):
    if not self.winners:
      raise ValueError(
        f'Area {self.name!r} does not have assembly; cannot fix.')
      return
    self.fixed_assembly = True

  def unfix_assembly(self):
    self.fixed_assembly = False

  def get_num_ever_fired(self):
    if self.explicit:
      return self.num_ever_fired 
    else:
      return self.w


class Brain:
  """A model brain.

  Attributes:
    area_by_name: Mapping from brain area-name tag to corresponding Area
      instance. (Original code: .areas).
    stimulus_size_by_name: Mapping from a stimulus-name to its number of
      neurons.
    connectomes_by_stimulus: Mapping from stimulus-name to a mapping
      from area-name to an activation-vector for that area.
      (Original code: .stimuli_connectomes)
    connectomes: Mapping from a 'source' area-name to a mapping from a
      'target' area-name to a [source_size, target_size]-bool-ndarray
      with connections. (TODO(tfish): Rename and replace with index-vector.)
      The source-index, respectively target-index, reference neurons in the
      "active assembly".
    p: Neuron connection-probability.
    save_size: Boolean flag, whether to save sizes.
    save_winners: Boolean flag, whether to save winners.
    disable_plasticity: Debug flag for disabling plasticity.
  """
  def __init__(self, p, save_size=True, save_winners=False, seed=0):
    self.area_by_name = {}
    self.stimulus_size_by_name = {}
    self.connectomes_by_stimulus = {}
    self.connectomes = {}
    self.p = p
    self.save_size = save_size
    self.save_winners = save_winners
    self.disable_plasticity = False
    self._rng = np.random.default_rng(seed=seed)    
    # For debugging purposes in applications (eg. language)
    self._use_normal_ppf = False

  def add_stimulus(self, stimulus_name, size):
    """Add a stimulus to the current instance.

    Args:
      stimulus_name: The name with which the stimulus will be registered.
      size: Number of firing neurons in this stimulus(?).
    """
    self.stimulus_size_by_name[stimulus_name] = size
    this_stimulus_connectomes = {}
    for area_name in self.area_by_name:
      if self.area_by_name[area_name].explicit:
        this_stimulus_connectomes[area_name] = self._rng.binomial(
            size, self.p,
            size=self.area_by_name[area_name].n).astype(np.float32)
      else:
        this_stimulus_connectomes[area_name] = np.empty(0, dtype=np.float32)
      self.area_by_name[area_name].beta_by_stimulus[stimulus_name] = (
        self.area_by_name[area_name].beta)
    self.connectomes_by_stimulus[stimulus_name] = this_stimulus_connectomes

  def add_area(self, area_name, n, k, beta):
    """Add a brain area to the current instance.

    Args:
      area_name: The name of the new area.
      n: Number of neurons.
      k: Number of that can fire in this area, at any time step.
      beta: default area-beta.
    """
    self.area_by_name[area_name] = the_area = Area(area_name, n, k, beta=beta)

    for stim_name, stim_connectomes in self.connectomes_by_stimulus.items():
      stim_connectomes[area_name] = np.empty(0, dtype=np.float32)
      the_area.beta_by_stimulus[stim_name] = beta

    new_connectomes = {}
    for other_area_name in self.area_by_name:
      other_area = self.area_by_name[other_area_name]
      other_area_size = other_area.n if other_area.explicit else 0
      new_connectomes[other_area_name] = np.empty((0, other_area_size), dtype=np.float32)
      if other_area_name != area_name:
        self.connectomes[other_area_name][area_name] = np.empty(
          (other_area_size, 0), dtype=np.float32)
      # by default use beta for plasticity of synapses from this area
      # to other areas
      # by default use other area's beta for synapses from other area
      # to this area
      other_area.beta_by_area[area_name] = other_area.beta
      the_area.beta_by_area[other_area_name] = beta
    self.connectomes[area_name] = new_connectomes

  def add_explicit_area(self,
                        area_name, n, k, beta, *,
                        custom_inner_p=None,
                        custom_out_p=None,
                        custom_in_p=None):
    """Add an explicit ('non-lazy') area to the instance.

    Args:
      area_name: The name of the new area.
      n: Number of neurons.
      k: Number of that can fire in this area, at any time step.
      beta: default area-beta.
      custom_inner_p: Optional self-linking probability.
      custom_out_p: Optional custom output-link probability.
      custom_in_p: Optional custom input-link probability.
    """
    # Explicitly set w to n so that all computations involving this area
    # are explicit.
    self.area_by_name[area_name] = the_area = Area(
        area_name, n, k, beta=beta, w=n, explicit=True)
    the_area.ever_fired = np.zeros(n, dtype=bool)
    the_area.num_ever_fired = 0

    for stim_name, stim_connectomes in self.connectomes_by_stimulus.items():
      stim_connectomes[area_name] = self._rng.binomial(
          self.stimulus_size_by_name[stim_name],
          self.p, size=n).astype(np.float32)
      the_area.beta_by_stimulus[stim_name] = beta

    inner_p = custom_inner_p if custom_inner_p is not None else self.p
    in_p = custom_in_p if custom_in_p is not None else self.p
    out_p = custom_out_p if custom_out_p is not None else self.p

    new_connectomes = {}
    for other_area_name in self.area_by_name:
      if other_area_name == area_name:  # create explicitly
        new_connectomes[other_area_name] = self._rng.binomial(
            1, inner_p, size=(n,n)).astype(np.float32)
      else:
        other_area = self.area_by_name[other_area_name]
        if other_area.explicit:
          other_n = self.area_by_name[other_area_name].n
          new_connectomes[other_area_name] = self._rng.binomial(
                  1, out_p, size=(n, other_n)).astype(np.float32)
          self.connectomes[other_area_name][area_name] = self._rng.binomial(
                  1, in_p, size=(other_n, n)).astype(np.float32)
        else: # we will fill these in on the fly
          # TODO: if explicit area added late, this will not work
          # But out_p to a non-explicit area must be default p,
          # for fast sampling to work.
          new_connectomes[other_area_name] = np.empty((n, 0), dtype=np.float32)
          self.connectomes[other_area_name][area_name] = np.empty((0, n), dtype=np.float32)
      self.area_by_name[other_area_name].beta_by_area[area_name] = (
        self.area_by_name[other_area_name].beta)
      self.area_by_name[area_name].beta_by_area[other_area_name] = beta
    self.connectomes[area_name] = new_connectomes

  def update_plasticity(self, from_area, to_area, new_beta):
    self.area_by_name[to_area].beta_by_area[from_area] = new_beta

  def update_plasticities(self,
                          area_update_map=EMPTY_MAPPING,
                          stim_update_map=EMPTY_MAPPING):
    # area_update_map consists of area1: list[ (area2, new_beta) ]
    # represents new plasticity FROM area2 INTO area1
    for to_area, update_rules in area_update_map.items():
      for from_area, new_beta in update_rules:
        self.update_plasticity(from_area, to_area, new_beta)

    # stim_update_map consists of area: list[ (stim, new_beta) ]f
    # represents new plasticity FROM stim INTO area
    for area, update_rules in stim_update_map.items():
      the_area = self.area_by_name[area]
      for stim, new_beta in update_rules:
        the_area.beta_by_stimulus[stim] = new_beta

  def activate(self, area_name, index):
    area = self.area_by_name[area_name]
    k = area.k
    assembly_start = k * index
    area.winners = list(range(assembly_start, assembly_start + k))
    area.fix_assembly()

  def project(self, areas_by_stim, dst_areas_by_src_area, verbose=0):
    # Validate stim_area, area_area well defined
    # areas_by_stim: {"stim1":["A"], "stim2":["C","A"]}
    # dst_areas_by_src_area: {"A":["A","B"],"C":["C","A"]}
    
    stim_in = collections.defaultdict(list)
    area_in = collections.defaultdict(list)

    for stim, areas in areas_by_stim.items():
      if stim not in self.stimulus_size_by_name:
        raise IndexError(f"Not in brain.stimulus_size_by_name: {stim}")
      for area_name in areas:
        if area_name not in self.area_by_name:
          raise IndexError(f"Not in brain.area_by_name: {area_name}")
        stim_in[area_name].append(stim)
    for from_area_name, to_area_names in dst_areas_by_src_area.items():
      if from_area_name not in self.area_by_name:
        raise IndexError(from_area + " not in brain.area_by_name")
      for to_area_name in to_area_names:
        if to_area_name not in self.area_by_name:
          raise IndexError(f"Not in brain.area_by_name: {to_area_name}")
        area_in[to_area_name].append(from_area_name)

    to_update_area_names = stim_in.keys() | area_in.keys()

    for area_name in to_update_area_names:
      area = self.area_by_name[area_name]
      num_first_winners = self.project_into(
        area, stim_in[area_name], area_in[area_name], verbose)
      area.num_first_winners = num_first_winners
      if self.save_winners:
        area.saved_winners.append(area._new_winners)

    # once everything is done, for each area in to_update: area.update_winners()
    for area_name in to_update_area_names:
      area = self.area_by_name[area_name]
      area._update_winners()
      if self.save_size:
        area.saved_w.append(area.w)

  def project_into(self, target_area, from_stimuli, from_areas, verbose=0):
    # projecting everything in from stim_in[area] and area_in[area]
    # calculate: inputs to self.connectomes[area] (previous winners)
    # calculate: potential new winners, Binomial(sum of in sizes, k-top)
    # k top of previous winners and potential new winners
    # if new winners > 0, redo connectome and intra_connectomes
    # have to wait to replace new_winners
    rng = self._rng
    area_by_name = self.area_by_name
    if verbose >= 1:
      print(f"Projecting {', '.join(from_stimuli)} "
            f" and {', '.join(from_areas)} into {target_area.name}")

    # If projecting from area with no assembly, complain.
    for from_area_name in from_areas:
      from_area = area_by_name[from_area_name]
      if not from_area.winners or from_area.w == 0:
        raise ValueError(f"Projecting from area with no assembly: {from_area}")

    # For experiments with a "fixed" assembly in some area.
    if target_area.fixed_assembly:
      target_area._new_winners = target_area.winners
      target_area._new_w = target_area.w
      first_winner_inputs = []
      num_first_winners_processed = 0

    else:
	    target_area_name = target_area.name
	    prev_winner_inputs = np.zeros(target_area.w, dtype=np.float32)
	    for stim in from_stimuli:
	      stim_inputs = self.connectomes_by_stimulus[stim][target_area_name]
	      prev_winner_inputs += stim_inputs
	    for from_area_name in from_areas:
	      connectome = self.connectomes[from_area_name][target_area_name]
	      for w in self.area_by_name[from_area_name].winners:
	        prev_winner_inputs += connectome[w]

	    if verbose >= 2:
	      print("prev_winner_inputs:", prev_winner_inputs)

	    # simulate area.k potential new winners if the area is not explicit
	    if not target_area.explicit:
	      input_size_by_from_area_index = []
	      num_inputs = 0
	      normal_approx_mean = 0.0
	      normal_approx_var = 0.0
	      for stim in from_stimuli:
	        local_k = self.stimulus_size_by_name[stim]
	        input_size_by_from_area_index.append(local_k)
	        num_inputs += 1
	        ### if self._use_normal_ppf:  # Not active currently.
	        ###   local_p = self.custom_stim_p[stim][target_area_name]
	        ###   normal_approx_mean += local_k * local_p
	        ###   normal_approx_var += ((local_k * local_p * (1 - local_p)) ** 2)
	      for from_area_name in from_areas:
	        # if self.area_by_name[from_area_name].w < self.area_by_name[from_area_name].k:
	        #   raise ValueError("Area " + from_area_name + "does not have enough support.")
	        effective_k = len(self.area_by_name[from_area_name].winners)
	        input_size_by_from_area_index.append(effective_k)
	        num_inputs += 1
	        ### if self._use_normal_ppf:  # Disabled for now.
	        ###   local_p = self.custom_stim_p[from_area_name][target_area_name]
	        ###   normal_approx_mean += effective_k * local_p
	        ###   normal_approx_var += ((effective_k * local_p * (1-p)) ** 2)

	      total_k = sum(input_size_by_from_area_index)
	      if verbose >= 2:
	        print(f"{total_k=} and {input_size_by_from_area_index=}")

	      effective_n = target_area.n - target_area.w
	      if effective_n <= target_area.k:
	      	raise RuntimeError(
	      	    f'Remaining size of area "{target_area_name}" too small to sample k new winners.')
	      # Threshold for inputs that are above (n-k)/n quantile.
	      quantile = (effective_n - target_area.k) / effective_n
	      if False:
	        pass
	      ### if self._use_normal_ppf:  # Disabled.
	      ###   # each normal approximation is N(n*p, n*p*(1-p))
	      ###   normal_approx_std = math.sqrt(normal_approx_var)
	      ###   alpha = binom.ppf(quantile, loc=normal_approx_mean,
	      ###                     scale=normal_approx_std)
	      else:
	        # self.p can be changed to have a custom connectivity into this
	        # brain area but all incoming areas' p must be the same
	        alpha = binom.ppf(quantile, total_k, self.p)
	      if verbose >= 2:
	        print(f"Alpha = {alpha}")
	      # use normal approximation, between alpha and total_k, round to integer
	      # create k potential_new_winners
	      if False:  # to update to: self._use_normal_ppf:
	        mu = normal_approx_mean
	        std = normal_approx_std
	      else:
	        mu = total_k * self.p
	        std = math.sqrt(total_k * self.p * (1.0 - self.p))
	      a = (alpha - mu) / std
	      b = (total_k - mu) / std
	      potential_new_winner_inputs = (mu + truncnorm.rvs(
	        a, b, scale=std, size=target_area.k)).round(0)

	      if verbose >= 2:
	        print(f"potential_new_winner_inputs: {potential_new_winner_inputs}")

	      # take max among prev_winner_inputs, potential_new_winner_inputs
	      # get num_first_winners (think something small)
	      # can generate area._new_winners, note the new indices
	      all_potential_winner_inputs = np.concatenate(
	          [prev_winner_inputs, potential_new_winner_inputs])
	    else:  # Case: Area is explicit.
	      all_potential_winner_inputs = prev_winner_inputs

	    new_winner_indices = heapq.nlargest(target_area.k,
	                                        range(len(all_potential_winner_inputs)),
	                                        all_potential_winner_inputs.__getitem__)
	    if target_area.explicit:
	        for winner in new_winner_indices:
	        	if not target_area.ever_fired[winner]:
	        		target_area.ever_fired[winner] = True 
	        		target_area.num_ever_fired += 1

	    num_first_winners_processed = 0

	    if not target_area.explicit:
	      first_winner_inputs = []
	      for i in range(target_area.k):
	        if new_winner_indices[i] >= target_area.w:
	          # Winner-index larger than `w` means that this winner was
	          # first-activated here.
	          first_winner_inputs.append(
	              all_potential_winner_inputs[new_winner_indices[i]])
	          new_winner_indices[i] = target_area.w + num_first_winners_processed
	          num_first_winners_processed += 1
	    target_area._new_winners = new_winner_indices
	    target_area._new_w = target_area.w + num_first_winners_processed

	    if verbose >= 2:
	      print(f"new_winners: {target_area._new_winners}")

	    # for i in num_first_winners
	    # generate where input came from
	      # 1) can sample input from array of size total_k, use ranges
	      # 2) can use stars/stripes method: if m total inputs,
	      #    sample (m-1) out of total_k
	    inputs_by_first_winner_index = [None] * num_first_winners_processed
	    for i in range(num_first_winners_processed):
	      input_indices = rng.choice(range(total_k),
	                                 int(first_winner_inputs[i]),
	                                 replace=False)
	      num_connections_by_input_index = np.zeros(num_inputs)
	      total_so_far = 0
	      for j in range(num_inputs):
	        num_connections_by_input_index[j] = sum(
	          total_so_far + input_size_by_from_area_index[j] > w >= total_so_far
	          for w in input_indices)
	        total_so_far += input_size_by_from_area_index[j]
	      inputs_by_first_winner_index[i] = num_connections_by_input_index
	      if verbose >= 2:
	        print(f"For first_winner # {i} with input "
	              f"{first_winner_inputs[i]} split as so: "
	              f"{num_connections_by_input_index}")

    # connectome for each stim->area
      # add num_first_winners_processed cells, sampled input * (1+beta)
      # for i in repeat_winners, stimulus_inputs[i] *= (1+beta)
    num_inputs_processed = 0
    for stim in from_stimuli:
      connectomes = self.connectomes_by_stimulus[stim]
      if num_first_winners_processed > 0:
        connectomes[target_area_name] = target_connectome = np.resize(
            connectomes[target_area_name],
            target_area._new_w)
      else:
        target_connectome = connectomes[target_area_name]
      first_winner_synapses = target_connectome[target_area.w:]
      for i in range(num_first_winners_processed):
        first_winner_synapses[i] = (
            inputs_by_first_winner_index[i][num_inputs_processed])
      stim_to_area_beta = target_area.beta_by_stimulus[stim]
      if self.disable_plasticity:
        stim_to_area_beta = 0.0
      for i in target_area._new_winners:
        target_connectome[i] *= 1 + stim_to_area_beta
      if verbose >= 2:
        print(f"{stim} now looks like: ")
        print(self.connectomes_by_stimulus[stim][target_area_name])
      num_inputs_processed += 1

    # update connectomes from stimuli that were not fired this round into the area.
    if (not target_area.explicit) and (num_first_winners_processed > 0):
	    for stim_name, connectomes in self.connectomes_by_stimulus.items():
	        if stim_name in from_stimuli:
	    	    continue
	        connectomes[target_area_name] = the_connectome = np.resize(
	    		connectomes[target_area_name],
	    		target_area._new_w)
	        the_connectome[target_area.w:] = rng.binomial(
                self.stimulus_size_by_name[stim_name], self.p,
                size=(num_first_winners_processed))

    # connectome for each in_area->area
      # add num_first_winners_processed columns
      # for each i in num_first_winners_processed, fill in (1+beta) for chosen neurons
      # for each i in repeat_winners, for j in in_area.winners, connectome[j][i] *= (1+beta)
    for from_area_name in from_areas:
      from_area_w = self.area_by_name[from_area_name].w
      from_area_winners = self.area_by_name[from_area_name].winners
      from_area_winners_set = set(from_area_winners)
      from_area_connectomes = self.connectomes[from_area_name]
      # Q: Can we replace .pad() with numpy.resize() here?
      the_connectome = from_area_connectomes[target_area_name] = np.pad(
          from_area_connectomes[target_area_name],
          ((0, 0), (0, num_first_winners_processed)))
      for i in range(num_first_winners_processed):
        total_in = inputs_by_first_winner_index[i][num_inputs_processed]
        sample_indices = rng.choice(from_area_winners, int(total_in), replace=False)
        for j in sample_indices:
          the_connectome[j, target_area.w + i] = 1.0
        for j in range(from_area_w):
          if j not in from_area_winners_set:
            the_connectome[j, target_area.w + i] = rng.binomial(1, self.p)
      area_to_area_beta = (
        0 if self.disable_plasticity
        else target_area.beta_by_area[from_area_name])
      for i in target_area._new_winners:
        for j in from_area_winners:
          the_connectome[j, i] *= 1.0 + area_to_area_beta
      if verbose >= 2:
        print(f"Connectome of {from_area_name} to {target_area_name} is now:",
              the_connectome)
      num_inputs_processed += 1

    # expand connectomes from other areas that did not fire into area
    # also expand connectome for area->other_area
    for other_area_name, other_area in self.area_by_name.items():
      other_area_connectomes = self.connectomes[other_area_name]
      if other_area_name not in from_areas:
        the_other_area_connectome = other_area_connectomes[target_area_name] = (
          np.pad(
              other_area_connectomes[target_area_name],
              ((0, 0), (0, num_first_winners_processed))))
        the_other_area_connectome[:, target_area.w:] = rng.binomial(
          1, self.p, size=(the_other_area_connectome.shape[0],
                           target_area._new_w - target_area.w))
      # add num_first_winners_processed rows, all bernoulli with probability p
      target_area_connectomes = self.connectomes[target_area_name]
      the_target_area_connectome = target_area_connectomes[other_area_name] = (
        np.pad(
          target_area_connectomes[other_area_name],
          ((0, num_first_winners_processed), (0, 0))))
      the_target_area_connectome[target_area.w:, :] = rng.binomial(
          1, self.p,
          size=(target_area._new_w - target_area.w,
                the_target_area_connectome.shape[1]))
      if verbose >= 2:
        print(f"Connectome of {target_area_name!r} to {other_area_name!r} "
              "is now:", self.connectomes[target_area_name][other_area_name])

    return num_first_winners_processed
