#! /usr/bin/python3.9
# word order learner with "intermediate" helper TPJ areas
import brain
import brain_util as bu
import numpy as np
import random

PHON = "PHON"

TPJ_agent = "TPJ_agent"
TPJ_patient = "TPJ_patient"
TPJ_action = "TPJ_action"

TPJ_agent_helper = "TPJ_agent_helper"
TPJ_patient_helper = "TPJ_patient_helper"
TPJ_action_helper = "TPJ_action_helper"

SYNTAX_subject = "SYNTAX_subject"
SYNTAX_object = "SYNTAX_object"
SYNTAX_verb = "SYNTAX_verb"

MOOD = "MOOD"

# import word_order_int as wo

class LearnBrain(brain.Brain):
	def __init__(self, p, EXPLICIT_k=100, NON_EXPLICIT_k=100, NON_EXPLICIT_n=100000, beta=0.06, previous_constituent_fire_rounds=2, training_fire_rounds=10, num_nouns=2, num_verbs=2, num_moods=1,
					mood_to_trans_word_order = {0: ["S","V","O"]}):
			brain.Brain.__init__(self, p)
			self.num_nouns = num_nouns 
			self.num_verbs = num_verbs 
			self.num_words = self.num_nouns + self.num_verbs
			self.num_moods = num_moods

			self.add_explicit_area(PHON, self.num_words*EXPLICIT_k, EXPLICIT_k, beta)
			self.add_explicit_area(MOOD, self.num_moods*EXPLICIT_k, EXPLICIT_k, beta)

			self.add_area(TPJ_agent, NON_EXPLICIT_n, NON_EXPLICIT_k, beta)
			self.add_area(TPJ_patient, NON_EXPLICIT_n, NON_EXPLICIT_k, beta)
			self.add_area(TPJ_action, NON_EXPLICIT_n, NON_EXPLICIT_k, beta)

			self.add_area(TPJ_agent_helper, NON_EXPLICIT_n, NON_EXPLICIT_k, beta)
			self.add_area(TPJ_patient_helper, NON_EXPLICIT_n, NON_EXPLICIT_k, beta)
			self.add_area(TPJ_action_helper, NON_EXPLICIT_n, NON_EXPLICIT_k, beta)

			self.add_area(SYNTAX_subject, NON_EXPLICIT_n, NON_EXPLICIT_k, beta)
			self.add_area(SYNTAX_object, NON_EXPLICIT_n, NON_EXPLICIT_k, beta)
			self.add_area(SYNTAX_verb, NON_EXPLICIT_n, NON_EXPLICIT_k, beta)

			# TODO: reimplement to use self.inhibited map (and list of existing fibers) to generate project maps
		#	self.inhibited = {}
		#	for area in [PHON, MOOD, TPJ_agent, TPJ_patient, TPJ_action, TPJ_agent_helper, TPJ_patient_helper, TPJ_action_helper, SYNTAX_subject, SYNTAX_object, SYNTAX_verb]:
		#		self.inhibited[area] = True

			self.mood_to_trans_word_order = mood_to_trans_word_order
			self.training_fire_rounds = training_fire_rounds
			self.previous_constituent_fire_rounds = previous_constituent_fire_rounds


	def activate_PHON_index(self, index):
		self.activate(PHON, index)


	def project_training(self, constituent, time_step, first_word=False, previous_constituent=None):
		# constituent is one of S, V, O
		# set that TPJ area to fire

		if constituent == "S":
			TPJ_area, TPJ_helper_area, SYNTAX_area = TPJ_agent, TPJ_agent_helper, SYNTAX_subject
			other_constituent_one, other_constituent_one_SYNTAX_area = "O", SYNTAX_object
			other_constituent_two, other_constituent_two_SYNTAX_area = "V", SYNTAX_verb 
		elif constituent == "O":
			TPJ_area, TPJ_helper_area, SYNTAX_area = TPJ_patient, TPJ_patient_helper, SYNTAX_object
			other_constituent_one, other_constituent_one_SYNTAX_area = "S", SYNTAX_subject
			other_constituent_two, other_constituent_two_SYNTAX_area = "V", SYNTAX_verb 
		elif constituent == "V":
			TPJ_area, TPJ_helper_area, SYNTAX_area = TPJ_action, TPJ_action_helper, SYNTAX_verb
			other_constituent_one, other_constituent_one_SYNTAX_area = "O", SYNTAX_object
			other_constituent_two, other_constituent_two_SYNTAX_area = "S", SYNTAX_subject

		project_map = {}
		project_map[PHON] = [TPJ_area]
		project_map[TPJ_area] = [TPJ_helper_area, TPJ_area]
		project_map[TPJ_helper_area] = [TPJ_helper_area, TPJ_area, SYNTAX_area]
		project_map[MOOD] = [SYNTAX_area]
		if time_step > 0:
			project_map[SYNTAX_area] = [SYNTAX_area]
		if first_word:
			project_map[MOOD] += [TPJ_helper_area]
		if time_step <= self.previous_constituent_fire_rounds:
			if previous_constituent == other_constituent_one:
				project_map[other_constituent_one_SYNTAX_area] = [TPJ_helper_area]
			elif previous_constituent == other_constituent_two: 
				project_map[other_constituent_two_SYNTAX_area] = [TPJ_helper_area]

		self.project({}, project_map)

	def activate_role(self, PHON_index, TPJ_area_name, TPJ_helper_area_name, num_firings):
		self.activate(PHON, PHON_index)
		self.project({}, {PHON: [TPJ_area_name]})
		self.project({}, {PHON: [TPJ_area_name], TPJ_area_name: [TPJ_helper_area_name, TPJ_area_name]})
		for _ in range(num_firings):
			self.project({}, {PHON: [TPJ_area_name], TPJ_area_name: [TPJ_helper_area_name, TPJ_area_name],
							TPJ_helper_area_name: [TPJ_area_name, TPJ_helper_area_name]})

	def input_random_trans_sentence(self, mood_index=None, num_tpj_firings=10):
		# generate a random noun for subject, i.e. PHON[0:self.num_nouns]
		subj_index = random.randint(0, self.num_nouns - 1)
		# generate a random noun for object, i.e. PHON[0:self.num_nouns]
		# this allows subject to be same as object at least in training
		obj_index = random.randint(0, self.num_nouns - 1)
		# generate a random verb, i.e. PHON[self.num_nouns:]
		verb_index = random.randint(self.num_nouns, self.num_words - 1)

		if not mood_index:
			mood_index = random.randint(0, self.num_moods-1)

		self.activate(MOOD, mood_index)
		
		self.activate_role(subj_index, TPJ_agent, TPJ_agent_helper, num_tpj_firings)
		self.activate_role(obj_index, TPJ_patient, TPJ_patient_helper, num_tpj_firings)
		self.activate_role(verb_index, TPJ_action, TPJ_action_helper, num_tpj_firings)

		trans_word_order = self.mood_to_trans_word_order[mood_index]

		for t in range(self.training_fire_rounds):
			self.project_training(trans_word_order[0], t, True, None)

		for t in range(self.training_fire_rounds):
			self.project_training(trans_word_order[1], t, False, trans_word_order[0])

		for t in range(self.training_fire_rounds):
			self.project_training(trans_word_order[2], t, False, trans_word_order[1])

	def train(self, num):
		for i in range(num):
			self.input_random_trans_sentence()
			print("Finished sentence ", i)

	def get_total_input(self, from_area, to_area):
		# assumes an active assembly in both from_area and to_area
		total_input = 0.0
		connectome = self.connectomes[from_area][to_area]
		for w in self.area_by_name[from_area].winners:
			for u in self.area_by_name[to_area].winners:
				total_input += connectome[w, u]
		return total_input

	def get_biggest_input_TPJ_from_mood(self):
		mood_to_agent = self.get_total_input(MOOD, TPJ_agent_helper)
		mood_to_patient = self.get_total_input(MOOD, TPJ_patient_helper)
		mood_to_action = self.get_total_input(MOOD, TPJ_action_helper)

		if (mood_to_agent > mood_to_patient) and (mood_to_agent > mood_to_action):
			return TPJ_agent_helper

		if (mood_to_patient > mood_to_agent) and (mood_to_patient > mood_to_action):
			return TPJ_patient_helper

		return TPJ_action_helper

	def get_biggest_input_TPJ_from_syntax(self, syntax_area_name):
		target_helper_areas = self.get_target_TPJ_areas(syntax_area_name)
		first_target_input = self.get_total_input(syntax_area_name, target_helper_areas[0])
		second_target_input = self.get_total_input(syntax_area_name, target_helper_areas[1])
		if first_target_input > second_target_input:
			return target_helper_areas[0]
		return target_helper_areas[1]

	def get_syntax_area(self, TPJ_helper_name):
		if TPJ_helper_name == TPJ_agent_helper:
			return SYNTAX_subject 
		if TPJ_helper_name == TPJ_patient_helper:
			return SYNTAX_object 
		if TPJ_helper_name == TPJ_action_helper:
			return SYNTAX_verb

	def get_target_TPJ_areas(self, syntax_area_name):
		if syntax_area_name == SYNTAX_subject: 
			return [TPJ_patient_helper, TPJ_action_helper]
		if syntax_area_name == SYNTAX_object:
			return [TPJ_agent_helper, TPJ_action_helper]
		if syntax_area_name == SYNTAX_verb:
			return [TPJ_agent_helper, TPJ_patient_helper]

	def helper_to_symbol(self, helper_area_name):
		if helper_area_name == TPJ_agent_helper:
			return "S"
		if helper_area_name == TPJ_patient_helper:
			return "O"
		if helper_area_name == TPJ_action_helper:
			return "V"

	def activate_TPJ_generation(self, PHON_index, TPJ_area_name, TPJ_helper_area_name, num_firings):
		self.activate(PHON, PHON_index)
		self.project({}, {PHON: [TPJ_area_name]})
		self.project({}, {PHON: [TPJ_area_name], TPJ_area_name: [TPJ_area_name, TPJ_helper_area_name]})
		for _ in range(num_firings):
			self.project({}, {PHON: [TPJ_area_name], TPJ_area_name: [TPJ_area_name, TPJ_helper_area_name],
				TPJ_helper_area_name: [TPJ_area_name, TPJ_helper_area_name]})

	def generate_random_sentence(self, mood_index=0, num_tpj_firings=3):
		# generate a random noun for subject, i.e. PHON[0:self.num_nouns]
		subj_index = random.randint(0, self.num_nouns - 1)
		# generate a random noun for object, i.e. PHON[0:self.num_nouns]
		obj_index = random.randint(0, self.num_nouns - 1)
		while obj_index == subj_index:
			obj_index = random.randint(0, self.num_nouns - 1)
		# generate a random verb, i.e. PHON[self.num_nouns:]
		verb_index = random.randint(self.num_nouns, self.num_words - 1)

		self.no_plasticity = True 
		self.activate(MOOD, mood_index)
		self.activate_TPJ_generation(subj_index, TPJ_agent, TPJ_agent_helper, num_tpj_firings)
		self.activate_TPJ_generation(obj_index, TPJ_patient, TPJ_patient_helper, num_tpj_firings)
		self.activate_TPJ_generation(verb_index, TPJ_action, TPJ_action_helper, num_tpj_firings)
		
		## fire from MOOD to the TPJ helper areas, the one with biggest input wins
		current_active_TPJ_helper = self.get_biggest_input_TPJ_from_mood()
		self.project({}, {MOOD: [current_active_TPJ_helper]})
		print("Next helper area is ", current_active_TPJ_helper)
		# if doing full version, also fire from helper to TPJ now
		order = [self.helper_to_symbol(current_active_TPJ_helper)]

		for _ in range(2):
			current_syntax_area = self.get_syntax_area(current_active_TPJ_helper)
			self.project({}, {current_active_TPJ_helper: [current_syntax_area], MOOD: [current_syntax_area]})
			# this line is where "magic happens" and we go to next constituent
			target_TPJ_areas = self.get_target_TPJ_areas(current_syntax_area)
			self.project({}, {current_syntax_area: target_TPJ_areas, TPJ_agent: [TPJ_agent_helper], TPJ_patient: [TPJ_patient_helper], TPJ_action: [TPJ_action_helper]})
			current_active_TPJ_helper = self.get_biggest_input_TPJ_from_syntax(current_syntax_area)
			print("Next helper area is ", current_active_TPJ_helper)
			order.append(self.helper_to_symbol(current_active_TPJ_helper))

		self.no_plasticity = False
		return order

# more rigorous / better approach
# first fire MOOD and all TPJ into all TPJ helpers:
#	 pretend project, really compute inputs, select biggest helper, then actually project from MOOD
# then repeat: 
# 	fire from non-inhibited helper to TPJ to PHON
#	fire from non-inhibited helper to syntax (note that by disinhibiting all syntax, we suddenly go to next word)
# 	pretend from syntax to all helper areas, select biggest helper, then actually project from syntax

