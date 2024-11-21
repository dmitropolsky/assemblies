import brain
import brain_util as bu
import numpy as np
import random

######## Example commands ########
# Word/part-of-speech acquisition experiments
# brain = learner.LearnBrain(0.05, LEX_k=100)
# brain.train_simple(30)
# brain.test_verb("RUN")
# brain.test_verb("JUMP")
# brain.test_verb("CAT")
# brain.test_word("CAT")

# example with "abstract" lexicon
# brain = learner.LearnBrain(0.05, LEX_k=100, num_nouns=3, num_verbs=3)
# brain.train(30)
# brain.testIndexedWord(0)
# brain.testIndexedWord(5)

# example with extra context areas
# brain = learner.LearnBrain(0.05, LEX_k=100, PHON_k=100, CONTEXTUAL_k=100, LEX_n=100000, extra_context_area_k=20, num_nouns=2, num_verbs=2, extra_context_model="C"", beta=0.06)

# Simple syntax experiment (skips individual word semantics acquisition)
# brain = learner.SimpleSyntaxBrain(0.1)
# brain.pre_train()
# brain.train("NV", train_interrogative=True)

DOG = "DOG"
CAT = "CAT"
JUMP = "JUMP"
RUN = "RUN"

# For bilingualism
LANG = "LANG"
PERRO = "PERRO"
GATO = "GATO"
SALTAR = "SALTAR"
CORRER = "CORRER"

# area names
PHON = "PHON"
MOTOR = "MOTOR"
VISUAL = "VISUAL"
NOUN = "NOUN" # this is LEX_NOUN
VERB = "VERB" # this is LEX_VERB
CORE = "CORE" # area for all "cores" (LRI populations)
NOUN_CORE_INDEX = 0
VERB_CORE_INDEX = 1
NOUN_CORE = "NOUN_CORE"
VERB_CORE = "VERB_CORE"
NOUN_VERB = "NOUN_VERB"

# syntax areas
SEQ = "SEQ"
MOOD = "MOOD"



PHON_INDICES = {
	DOG: 0,
	CAT: 1,
	JUMP: 2,
	RUN: 3,
	PERRO: 4,
	GATO: 5,
	SALTAR: 6,
	CORRER: 7
}

# lexicon_sizes_experiment(2, 10, p=0.05, LEX_k=50, LEX_n=100000, beta=0.06, repeat=5, output_file="lex_size.txt")
# lexicon_sizes_experiment(2, 6, p=0.05, LEX_k=50, LEX_n=100000, beta=0.06, repeat=5, extra_context_model="C", extra_context_area_k=20, output_file="lex_size_extracontext.txt")
def lexicon_sizes_experiment(start, end, p=0.05, LEX_k=50, LEX_n=100000, beta=0.06, extra_context_areas=0, extra_context_model="B", use_extra_context=False, extra_context_area_k=20, repeat=1, output_file=None):
	results = {}
	if output_file:
		f = open(output_file, 'a')
	for n in range(start, end+1):
		results[n] = []
		if output_file:
			f.write(str(n)+",")
		for _ in range(repeat):
			brain = LearnBrain(p, LEX_k=LEX_k, LEX_n=LEX_n, num_nouns=n, num_verbs=n, beta=beta, extra_context_areas=extra_context_areas, extra_context_model=extra_context_model)
			brain.no_print=True
			num_sentences_needed = brain.train_experiment_randomized(use_extra_context=use_extra_context)
			results[n].append(num_sentences_needed)
			if output_file:
				f.write(str(num_sentences_needed)+",")
		if output_file:
			f.write("\n")
	if output_file:
		f.close()
	return results

# betas_experiment(0.1, 0.05, 0.01)
# betas_experiment(0.1, 0.015, 0.005, p=0.05, LEX_k=50, LEX_n=100000, num_nouns=2, num_verbs=2, repeat=5, output_file="betas.txt")
# betas_experiment(0.1, 0.095, 0.005, p=0.05, LEX_k=50, LEX_n=100000, num_nouns=2, num_verbs=2, repeat=2, output_file="TEST_betas.txt")
def betas_experiment(start, end, decrement, p=0.05, LEX_k=50, LEX_n=100000, num_nouns=2, num_verbs=2, repeat=1, output_file=None):
	results = {}
	if output_file:
		f = open(output_file, 'a')

	beta = start 
	while beta >= end: 
		results[beta] = []
		if output_file:
			f.write(str(beta)+",")
		for _ in range(repeat):
			brain = LearnBrain(p, LEX_k=LEX_k, LEX_n=LEX_n, num_nouns=num_nouns, num_verbs=num_verbs, beta=beta)
			brain.no_print=True
			num_sentences_needed = brain.train_experiment_randomized()
			results[beta].append(num_sentences_needed)
			print(str(beta) + ": " + str(num_sentences_needed))
			if output_file:
				f.write(str(num_sentences_needed)+",")
		beta -= decrement
		if output_file:
			f.write("\n")
	if output_file:
		f.close()
	return results

# p_experiment(0.01, 0.05, 0.01)
def p_experiment(start, end, increment, LEX_k=50, LEX_n=100000, CONTEXTUAL_k=100, PHON_k=100, beta=0.05, num_nouns=2, num_verbs=2):
	results = {}
	p = start 
	while p <= end:
		brain = LearnBrain(p, LEX_k=LEX_k, LEX_n=LEX_n, num_nouns=num_nouns, num_verbs=num_verbs, beta=beta)
		brain.no_print=True
		num_sentences_needed = brain.train_experiment_randomized(increment=5)
		results[p] = num_sentences_needed
		print(str(p) + ": " + str(num_sentences_needed))
		p += increment 
	return results

# this function returns num WORDS used for training.. i.e. project rounds -= num words * brain.proj_rounds
# to compare to regular training, num words = 2 * num_sentences_needed
# single_word_tutoring_exp(0.1, 0.015, 0.005, p=0.05, LEX_k=50, LEX_n=100000, num_nouns=2, num_verbs=2, repeat=5, output_file="betas.txt")
# single_word_tutoring_exp(2, 3, p=0.05, LEX_k=50, LEX_n=100000, single_word_frequency=2, repeat=1, output_file="TEST_single.txt")
def single_word_tutoring_exp(lex_size_start, lex_size_end, p=0.05, LEX_k=50, LEX_n=100000, beta=0.06, single_word_frequency=2, testing_increment=1,  repeat=1, output_file=None):
	results = {}
	if output_file:
		f = open(output_file, 'a')

	lex_size = lex_size_start
	while lex_size <= lex_size_end:
		results[lex_size] = []
		if output_file:
			f.write(str(lex_size)+",")
		for _ in range(repeat):
			brain = LearnBrain(p, LEX_k=LEX_k, LEX_n=LEX_n, num_nouns=lex_size, num_verbs=lex_size, beta=beta)
			brain.no_print=True
			num_words_needed = brain.train_experiment_randomized_with_tutoring(single_word_frequency=single_word_frequency, testing_increment=testing_increment)
			results[lex_size].append(num_words_needed)
			if output_file:
				f.write(str(num_words_needed)+",")
			print(str(lex_size) + ": " + str(num_words_needed))
		lex_size += 1
		if output_file:
			f.write("\n")
	if output_file:
		f.close()
	return results


class LearnBrain(brain.Brain):
	def __init__(self, p, PHON_k=100, CONTEXTUAL_k=100, EXPLICIT_k=100, LEX_k=100, LEX_n=10000, beta=0.06, proj_rounds=2,
		CORE_k=10, bilingual=False, LANG_k=100, num_nouns=2, num_verbs=2, extra_context_areas=0, extra_context_area_k=10, extra_context_model="B", extra_context_delay=0):
		brain.Brain.__init__(self, p)
		self.bilingual = bilingual

		# make this sum of #verbs + #nouns, more easily adjustable
		self.num_nouns = num_nouns
		self.num_verbs = num_verbs
		self.lex_size = self.num_nouns + self.num_verbs
		self.extra_context_areas = extra_context_areas
		self.extra_context_model = extra_context_model
		self.extra_context_delay = extra_context_delay
		# TODO: bilingualism
		# if self.bilingual:
		#	self.lex_size *= 2
		self.add_explicit_area(PHON, self.lex_size*PHON_k, PHON_k, beta)
		self.add_explicit_area(MOTOR, self.num_verbs*CONTEXTUAL_k, CONTEXTUAL_k, beta)
		self.add_explicit_area(VISUAL, self.num_nouns*CONTEXTUAL_k, CONTEXTUAL_k, beta)
		self.sentences_parsed = 0 

		if self.extra_context_model == "B" and self.extra_context_areas > 0:
			for i in range(extra_context_areas):
				extra_context_area_name = self.get_extra_context_area_name(i) 
				self.add_explicit_area(extra_context_area_name, self.lex_size*extra_context_area_k, extra_context_area_k, beta)
			self.extra_context_map = {}
			for word_index in range(self.lex_size):
				self.extra_context_map[word_index] = random.randint(0, self.extra_context_areas-1)

		if self.extra_context_model == "C":
			self.extra_context_areas = self.lex_size
			for i in range(self.extra_context_areas):
				extra_context_area_name = self.get_extra_context_area_name(i) 
				self.add_explicit_area(extra_context_area_name, extra_context_area_k, extra_context_area_k, beta)
			self.extra_context_map = {}
			for word_index in range(self.lex_size):
				self.extra_context_map[word_index] = word_index

		self.add_area(NOUN, LEX_n, LEX_k, beta)
		self.add_area(VERB, LEX_n, LEX_k, beta)
		self.proj_rounds = proj_rounds

		if self.bilingual:
			self.add_explicit_area(LANG, 2*LANG_k, LANG_k, beta)

	def tutor_single_word(self, word):
		self.activate_context(word)
		self.activate_PHON(word)
		if self.bilingual:
			self.activate_lang(word)
		self.project_star(mutual_inhibition=True)

	def tutor_single_indexed_word(self, word_index):
		self.activate_index_context(word_index)
		self.activate(PHON, word_index)
		self.project_star(mutual_inhibition=True)

	def tutor_random_word(self):
		word_index = random.randint(0, self.lex_size-1)
		self.tutor_single_indexed_word(word_index)

	def get_extra_context_area_name(self, index):
		return "CONTEXT_" + str(index)

	def activate_lang(self, word):
		lang_index = 0 if (word in [CAT, DOG, JUMP, RUN]) else 1
		print("GOING TO ACTIVATE IN LANG INDEX " + str(lang_index))
		self.activate(LANG, lang_index)

	def activate_context(self, word):
		if word == DOG or word == PERRO:
			self.activate(VISUAL, 0)
		if word == CAT or word == GATO:
			self.activate(VISUAL, 1)
		if word == JUMP or word == SALTAR:
			self.activate(MOTOR, 0)
		if word == RUN or word == CORRER:
			self.activate(MOTOR, 1)

	def activate_index_context(self, word_index, activate_extra_context=False):
		if activate_extra_context:
			extra_context_index = self.extra_context_map[word_index]
			extra_context_area_name = self.get_extra_context_area_name(extra_context_index)
			self.activate(extra_context_area_name, word_index)
		if word_index < self.num_nouns:
			self.activate(VISUAL, word_index)
		else:
			motor_index = word_index - self.num_nouns
			self.activate(MOTOR, motor_index)

	def get_context_area(self, word):
		if word in [CAT, DOG, PERRO, GATO]:
			return VISUAL
		elif word in [JUMP, RUN, SALTAR, CORRER]:
			return MOTOR
		return None

	def get_index_context_area(self, word_index):
		if word_index < self.num_nouns:
			return VISUAL
		return MOTOR

	def get_index_lexical_area(self, word_index):
		if word_index < self.num_nouns:
			return NOUN
		return VERB

	def activate_PHON(self, word):
		self.activate(PHON, PHON_INDICES[word])

	def project_star(self, mutual_inhibition=False):
		# compute the initial (t=1) project map; NOUN and VERB do not yet have any winners
		project_map = {PHON: [NOUN, VERB]}
		if self.area_by_name[MOTOR].winners:
			project_map[MOTOR] = [VERB]
		if self.area_by_name[VISUAL].winners:
			project_map[VISUAL] = [NOUN]
		if self.bilingual:
			project_map[LANG] = [NOUN, VERB]
		if self.extra_context_areas:
			for i in range(self.extra_context_areas):
				extra_context_area_name = self.get_extra_context_area_name(i)
				if self.area_by_name[extra_context_area_name].winners:
					project_map[extra_context_area_name] = [NOUN, VERB]
					print("PROJECTING FROM EXTRA AREA " + str(extra_context_area_name))
		self.project({}, project_map)


		# for subsequent rounds, now include recurrent firing + firing from NOUN/VERB
		project_map[NOUN] = [PHON, NOUN]
		project_map[VERB] = [PHON, VERB]
		if self.area_by_name[MOTOR].winners:
			project_map[VERB] += [MOTOR]
		if self.area_by_name[VISUAL].winners:
			project_map[NOUN] += [VISUAL]

		if mutual_inhibition:
			noun_input = self.get_total_input(NOUN)
			verb_input = self.get_total_input(VERB)
			if noun_input > verb_input:
				del project_map[VERB]
				project_map[PHON].remove(VERB)
				if self.bilingual:
					project_map[LANG].remove(VERB)
			else:
				del project_map[NOUN]
				project_map[PHON].remove(NOUN)
				if self.bilingual:
					project_map[LANG].remove(NOUN)

		for _ in range(self.proj_rounds):
			self.project({}, project_map)

	def parse_sentence(self, sentence):
		# sentence in the form [NOUN verb]
		for word in sentence:
			self.activate_context(word)
		for word in sentence:
			self.activate_PHON(word)
			self.project_star()
		self.sentences_parsed += 1

	def parse_indexed_sentence(self, noun_index, verb_index, order="NV"):
		motor_index = verb_index - self.num_nouns
		self.activate(VISUAL, noun_index)
		self.activate(MOTOR, motor_index)
		if self.extra_context_areas > 0 and self.sentences_parsed > self.extra_context_delay:
			if self.extra_context_model == "A":
				for i in range(self.extra_context_areas):
					extra_context_area_name = self.get_extra_context_area_name(i)
					# randomly choose whether the verb or noun's context fires for this area
					if random.getrandbits(1):
						self.activate(extra_context_area_name, noun_index)
					else:
						self.activate(extra_context_area_name, verb_index)
			elif self.extra_context_model == "B":
				noun_extra_context_index = self.extra_context_map[noun_index]
				noun_extra_context_area_name = self.get_extra_context_area_name(noun_extra_context_index)
				verb_extra_context_index = self.extra_context_map[verb_index]
				verb_extra_context_area_name = self.get_extra_context_area_name(verb_extra_context_index)
				if noun_extra_context_index == verb_extra_context_index:
					if random.getrandbits(1):
						self.activate(noun_extra_context_area_name, noun_index)
					else:
						self.activate(noun_extra_context_area_name, verb_index)
				else:
					self.activate(noun_extra_context_area_name, noun_index)
					self.activate(verb_extra_context_area_name, verb_index)
			if self.extra_context_model == "C":
				noun_extra_context_area_name = self.get_extra_context_area_name(noun_index)
				verb_extra_context_area_name = self.get_extra_context_area_name(verb_index)
				self.activate(noun_extra_context_area_name, 0)
				self.activate(verb_extra_context_area_name, 0)
		if order == "NV":
			self.activate(PHON, noun_index)
		else:
			self.activate(PHON, verb_index)
		self.project_star()
		if order == "NV":
			self.activate(PHON, verb_index)
		else: 
			self.activate(PHON, noun_index)
		self.project_star()
		self.clear_context_winners()
		self.sentences_parsed += 1

	def clear_context_winners(self):
		self.area_by_name[VISUAL].winners = []
		self.area_by_name[MOTOR].winners = []
		if self.extra_context_areas > 0:
			for i in range(self.extra_context_areas):
				extra_context_area_name = self.get_extra_context_area_name(i)
				self.area_by_name[extra_context_area_name].winners = []

	def train_simple(self, rounds):
		sentence_1 = [CAT, JUMP]
		sentence_2 = [CAT, RUN]
		sentence_3 = [DOG, JUMP]
		sentence_4 = [DOG, RUN]
		sentences = [sentence_1, sentence_2, sentence_3, sentence_4]
		for i in range(rounds):
			print("Round " + str(i))
			for sentence in sentences:
				self.parse_sentence(sentence)

	def train_random_sentence(self):
		noun_index = random.randint(0, self.num_nouns-1)
		verb_index = random.randint(self.num_nouns, self.num_nouns + self.num_verbs-1)
		self.parse_indexed_sentence(noun_index, verb_index)

	def train_each_sentence(self):
		for noun_index in range(self.num_nouns):
			for verb_index in range(self.num_nouns, self.num_nouns + self.num_verbs):
				self.parse_indexed_sentence(noun_index, verb_index)

	def train(self, rounds):
		for i in range(rounds):
			print("Round " + str(i))
			self.train_each_sentence()

	def train_experiment(self, max_rounds=100, use_extra_context=False):
		for i in range(max_rounds):
			print("Round " + str(i))
			self.train_each_sentence()
			if self.test_all_words(use_extra_context=use_extra_context):
				print("Succeeded after " + str(i) + " rounds of all sentences.")
				return i  
		print("Did not succeed after " + str(max_rounds) + " rounds.")
		return None

# l = learner.LearnBrain(0.05, LEX_k=50, LEX_n=100000, num_nouns=2, num_verbs=2, beta=0.06)
# train_experiment_randomized()
	def train_experiment_randomized(self, max_samples=500, increment=1, start_testing=0, use_extra_context=False):
		#self.extra_context_areas = 0
		for i in range(0, max_samples):
			self.train_random_sentence()
			if (i > start_testing) and (i % increment == 0) and self.test_all_words(use_extra_context=use_extra_context):
				print("Succeeded after " + str(i) + " random sentences.")
				return i  
			#if i == 30:
			#	self.extra_context_areas = 2
		print("Did not succeed after " + str(max_samples) + " samples.")
		return None

	def train_experiment_randomized_with_tutoring(self, max_samples=500, testing_increment=1, start_testing=0, single_word_frequency=5):
		num_words = 0
		for i in range(1, max_samples+1):
			if (i % single_word_frequency) == 0:
				self.tutor_random_word()
				num_words += 1
			else:
				self.train_random_sentence()
				num_words += 2
			if (i > start_testing) and (i % testing_increment == 0) and self.test_all_words():
				print("Succeeded after " + str(num_words) + " random words.")
				return num_words 
		print("Did not succeed after " + str(max_samples) + " samples.")
		return None

	# property P test
	def test_all_words(self, use_extra_context=False):
		for word_index in range(self.num_nouns + self.num_verbs):
			if (self.testIndexedWord(word_index, use_extra_context=use_extra_context, no_print=True) != word_index):
				return False
		return True

	def get_explicit_assembly(self, area_name, min_overlap=0.75):
		if not self.area_by_name[area_name].winners:
			raise Exception("Cannot get word because no assembly in " + area_name)
		winners = set(self.area_by_name[area_name].winners)
		area = self.area_by_name[area_name]
		area_k = area.k
		threshold = min_overlap * area_k
		num_assemblies = int(area.n / area.k)
		for index in range(num_assemblies):
			assembly_start = index * area_k
			assembly = set(range(assembly_start, assembly_start + area_k))
			if len((winners & assembly)) >= threshold:
				return index 
		print("Got non-assembly in " + area_name)
		return None

	def get_PHON(self, min_overlap=0.75):
		index = self.get_explicit_assembly(PHON, min_overlap)
		for word, i in PHON_INDICES.items():
			if i == index:
				return word

	def test_verb(self, word, min_overlap=0.75):
		self.disable_plasticity = True
		self.area_by_name[PHON].unfix_assembly()
		self.activate_context(word)
		if self.bilingual:
			self.activate_lang(word)
		area = self.get_context_area(word)
		first_proj_map = {area: [VERB]}
		if self.bilingual:
			first_proj_map[LANG] = [VERB]
		self.project({}, first_proj_map)
		self.project({}, {VERB: [PHON]})
		self.disable_plasticity = False
		return self.get_PHON(min_overlap)

	def testIndexedWord(self, word_index, min_overlap=0.75, use_extra_context=False, no_print=False):
		self.disable_plasticity = True
		self.area_by_name[PHON].unfix_assembly()
		self.activate_index_context(word_index, use_extra_context)
		area = self.get_index_context_area(word_index)
		to_area = self.get_index_lexical_area(word_index)
		self.project({}, {area: [to_area]})
		self.project({}, {to_area: [PHON]})
		self.disable_plasticity = False
		out = self.get_explicit_assembly(PHON, min_overlap)
		if not no_print:
			print("For word " + str(word_index) + " got output " + str(out))
		self.clear_context_winners()
		self.disable_plasticity = False
		return out

	def test_noun(self, word, min_overlap=0.75):
		self.disable_plasticity = True
		self.area_by_name[PHON].unfix_assembly()
		self.activate_context(word)
		if self.bilingual:
			self.activate_lang(word)
		area = self.get_context_area(word)
		first_proj_map = {area: [NOUN]}
		if self.bilingual:
			first_proj_map[LANG] = [NOUN]
		self.project({}, first_proj_map)
		self.project({}, {NOUN: [PHON]})
		self.disable_plasticity = False
		return self.get_PHON(min_overlap)

	def get_input_from(self, from_area, to_area):
		from_winner_indices = self.area_by_name[from_area].winners
		to_winner_indices = self.area_by_name[to_area].winners 
		if (not from_winner_indices) or (not to_winner_indices):
			return 0
		total_input = ((self.connectomes[from_area][to_area])[from_winner_indices][:,to_winner_indices]).sum()
		return total_input

	def get_total_input(self, area):
		total_input = self.get_input_from(PHON, area)
		if area == NOUN:
			# also include VISUAL -> NOUN
			total_input += self.get_input_from(VISUAL, area)
		elif area == VERB:
			total_input += self.get_input_from(MOTOR, area)
		return total_input

	# property Q test
	def test_word(self, word):
		self.disable_plasticity = True
		self.activate_PHON(word)
		self.project({}, {PHON: [VERB, NOUN]})
		# compare inputs into both 

		print("Computing total synaptic inputs from PHON -> NOUN and VERB...")
		verb_sum = self.get_input_from(PHON, VERB)

		noun_sum = self.get_input_from(PHON, NOUN)

		print("Got input into VERB = " + str(verb_sum))
		print("Got input into NOUN = " + str(noun_sum))

		print("Firing NOUN and VERB recurrently, and computing overlap of winners at (t+1) with t")
		noun_winners = self.area_by_name[NOUN].winners[:]
		verb_winners = self.area_by_name[VERB].winners[:]
		self.project({}, {"NOUN": ["NOUN"], "VERB": ["VERB"]})
		noun_overlap = bu.overlap(noun_winners, self.area_by_name[NOUN].winners)
		verb_overlap = bu.overlap(verb_winners, self.area_by_name[VERB].winners)
		print("In NOUN: got " + str(noun_overlap) + " / " + str(len(noun_winners)) + " overlap.")
		print("In VERB: got " + str(verb_overlap) + " / " + str(len(verb_winners)) + " overlap.")
		self.disable_plasticity = False


# a brain that assumes single word representations have been learnt (stored in a combined LEX area called NOUN_VERB)
# and learns 2-word sentence (subject + intransitive verb) sentence word order, including with several moods with different word orders
# uses the SEQ area mechanism for learning word order statistics
class SimpleSyntaxBrain(brain.Brain):
	def __init__(self, p, CONTEXTUAL_k=100, EXPLICIT_k=100, beta=0.06, LEX_n=10000, LEX_k=100, proj_rounds=2, CORE_k=10):
		brain.Brain__init__(self, p)
		# Q: Do we need to "rewire" inside these areas (make the explicit assemblies more highly connected?)
		self.add_explicit_area(NOUN_VERB, 4*EXPLICIT_k, EXPLICIT_k, beta)
		# self.add_explicit_area(VERB, 2*EXPLICIT_k, EXPLICIT_k, beta)
		self.add_explicit_area(MOTOR, 2*CONTEXTUAL_k, CONTEXTUAL_k, beta)
		self.add_explicit_area(VISUAL, 2*CONTEXTUAL_k, CONTEXTUAL_k, beta)
		self.add_area(SEQ, LEX_n, LEX_k, beta)
		self.add_explicit_area(MOOD, 2*EXPLICIT_k, EXPLICIT_k ,beta)
		self.add_cores(CORE_k=CORE_k)
		self.proj_rounds = proj_rounds	

	def add_cores(self, CORE_k=10, CORE_inner_beta=0.05, CORE_outer_beta=0.1):
		# as of now cores only work with *explicit areas* (NOUN and VERB must be modelled as explicit)
		# TODO: rework brain.py to fill in connectomes from new explicit areas to used non-explicit areas
		# NOTE: high custom_in_p might not be needed depeneding how cores are acquired 
		# for some applications, custom_inner_p should be high 

		# CORE[0] = noun core
		# CORE[1] = verb core
		self.add_explicit_area(CORE, 2*CORE_k, CORE_k, CORE_inner_beta, 
			custom_inner_p=0.9, custom_out_p=0.9, custom_in_p=0.9)
		self.update_plasticity(CORE, NOUN_VERB, CORE_outer_beta)

	def parse(self, sentence, mood_state=0):
		# SEQ always projects to both cores, has to project to a core when it is also activated
		# CORES are in mutual inhibition... one with more input fires
		# CAT JUMP
		# activate MOOD[INDICATIVE] -> SEQ
		# activate NOUN[CAT] -> NOUN_CORE (at same time SEQ -> CORES)
		# NOUN_CORE -> SEQ 
		# activate VERB[JUMP] -> VERB_CORE (at same time SEQ -> CORES)

		self.activate(MOOD, mood_state)
		self.project({}, {MOOD: [SEQ]})
		for _ in range(self.proj_rounds):
			self.project({}, {MOOD: [SEQ], SEQ: [SEQ]})

		# key idea: we only want cores interacting with SEQ (so SEQ <-> cores only)
		for word in sentence:
			self.activate(NOUN_VERB, word)
			area_firing_into_core = NOUN_VERB
			self.project({}, {SEQ: [CORE, SEQ], area_firing_into_core: [CORE]})
			# Could capture more complexity by projecting CORE,SEQ -> SEQ (can hold some state this way)
			self.project({}, {CORE: [SEQ]})
			for _ in range(self.proj_rounds):
				self.project({}, {CORE: [SEQ], SEQ: [SEQ]})

	def pre_train(self, proj_rounds=20):
		# GOAL 1: connect word assemblies in {NOUN, VERB} with context assemblies (bidirectional)
		# GOAL 2: connect word assemblies in {NOUN, VERB} with their core (at least core -> word, maybe bidirectional)
		for noun_index in [0, 1]:
			self.activate(NOUN_VERB, noun_index)
			self.activate(VISUAL, noun_index)
			self.activate(CORE, NOUN_CORE_INDEX)
			# all assemblies in below project area fixed; this is just to ramp up edge weights via plasticity
			for _ in range(proj_rounds):
				self.project({}, {NOUN_VERB: [NOUN_VERB, VISUAL, CORE], VISUAL: [NOUN_VERB], CORE: [NOUN_VERB]})

		for verb_index in [2, 3]:
			self.activate(NOUN_VERB, verb_index)
			self.activate(MOTOR, verb_index - 2)
			self.activate(CORE, VERB_CORE_INDEX)
			for _ in range(proj_rounds):
				self.project({}, {NOUN_VERB: [NOUN_VERB, MOTOR, CORE], MOTOR: [NOUN_VERB], CORE: [NOUN_VERB]})

		self.area_by_name[CORE].unfix_assembly()
		self.area_by_name[NOUN_VERB].unfix_assembly()

	def pre_train_test(self):
		self.disable_plasticity = True
		self.area_by_name[CORE].unfix_assembly()
		for i in [0, 1]:
			self.activate(NOUN_VERB, i)
			self.project({}, {NOUN_VERB: [CORE]})
			out = self.get_explicit_assembly(CORE, min_overlap=0.9)
			if out != NOUN_CORE_INDEX:
				print("ERROR: a NOUN activated the VERB core")
				return
		for i in [2, 3]:
			self.activate(NOUN_VERB, i)
			self.project({}, {NOUN_VERB: [CORE]})
			out = self.get_explicit_assembly(CORE, min_overlap=0.9)
			if out != VERB_CORE_INDEX:
				print("ERROR: a VERB activated the NOUN core")
				return
		print("Passed tests from NOUN, VERB -> CORE")
		self.area_by_name[NOUN_VERB].unfix_assembly()
		self.activate(CORE, NOUN_CORE_INDEX)
		self.project({}, {CORE: [NOUN_VERB]})
		if self.get_explicit_assembly(NOUN_VERB, min_overlap=0.75):
			print("ERROR: projecting noun core -> NOUN, VERB gave explicit assembly")
			return 
		max_winner = max(self.area_by_name[NOUN_VERB].winners)
		if  max_winner >= (2 * self.area_by_name[NOUN_VERB].k):
			print("ERROR: proecting noun core -> NOUN, VERB yielded winner in verb part")
		print("Passed noun core -> noun verb, max winner was " + str(max_winner))
		self.activate(CORE, VERB_CORE_INDEX)
		self.project({}, {CORE: [NOUN_VERB]})
		if self.get_explicit_assembly(NOUN_VERB, min_overlap=0.75):
			print("ERROR: projecting noun core -> NOUN, VERB gave explicit assembly")
			return 
		min_winner = min(self.area_by_name[NOUN_VERB].winners)
		if  min_winner < (2 * self.area_by_name[NOUN_VERB].k):
			print("ERROR: proecting noun core -> NOUN, VERB yielded winner in verb part")
		print("Passed verb core -> noun verb, min winner was " + str(min_winner))

		self.disable_plasticity = False
		self.area_by_name[CORE].unfix_assembly()
		self.area_by_name[NOUN_VERB].unfix_assembly()


	# an experiment that trains the brain with 2 word sentences, possibly with 2 different moods / word orders
	def train(self, order, train_rounds=40, train_interrogative=False):
		if order == "NV":
			sentence = [0, 2]
		elif order == "VN":
			sentence = [2, 0]
		else:
			print("first argument must be NV or VN")
			return
		interrogative_sentence = sentence[:]
		interrogative_sentence.reverse()
		self.disable_plasticity = False
		for _ in range(train_rounds):
			self.parse(sentence, mood_state=0)
			if train_interrogative:
				self.parse(interrogative_sentence, mood_state=1)

		# Test
		self.disable_plasticity = True
		self.area_by_name[CORE].unfix_assembly()
		self.activate(MOOD, 0)
		self.project({}, {MOOD: [SEQ]})
		self.project({}, {SEQ: [CORE]})
		core_i = self.get_explicit_assembly(CORE)
		if core_i == NOUN_CORE_INDEX:
			print("First word is a NOUN")
		if core_i == VERB_CORE_INDEX:
			print("First word is a VERB")
		self.project({}, {CORE: [SEQ]})
		self.project({}, {SEQ: [CORE]})
		core_i = self.get_explicit_assembly(CORE)
		if core_i == NOUN_CORE_INDEX:
			print("Second word is a NOUN")
		if core_i == VERB_CORE_INDEX:
			print("Second word is a VERB")
		
		if train_interrogative:
			print("Now testing INTERROGATIVE word order...")
			self.area_by_name[CORE].unfix_assembly()
			self.activate(MOOD, 1)
			self.project({}, {MOOD: [SEQ]})
			self.project({}, {SEQ: [CORE]})
			core_i = self.get_explicit_assembly(CORE)
			if core_i == NOUN_CORE_INDEX:
				print("First word is a NOUN")
			if core_i == VERB_CORE_INDEX:
				print("First word is a VERB")
			self.project({}, {CORE: [SEQ]})
			self.project({}, {SEQ: [CORE]})
			core_i = self.get_explicit_assembly(CORE)
			if core_i == NOUN_CORE_INDEX:
				print("Second word is a NOUN")
			if core_i == VERB_CORE_INDEX:
				print("Second word is a VERB")

class LearnBrain_SimpleSyntax(LearnBrain):
	def __init__(self, p, PHON_k=100, CONTEXTUAL_k=100, EXPLICIT_k=100, LEX_k=100, LEX_n=10000, beta=0.06, proj_rounds=2,
		CORE_k=10, bilingual=False, LANG_k=100, num_nouns=2, num_verbs=2, extra_context_areas=0, extra_context_area_k=10, extra_context_model="B", extra_context_delay=0):
		super().__init__(p, PHON_k=100, CONTEXTUAL_k=100, EXPLICIT_k=100, LEX_k=100, LEX_n=10000, beta=0.06, proj_rounds=2,
			CORE_k=10, bilingual=False, LANG_k=100, num_nouns=2, num_verbs=2, extra_context_areas=0, extra_context_area_k=10, extra_context_model="B", extra_context_delay=0)
		# note that custom_out_p only works into *other explicit areas*
		# that means CORE -> NOUN/VERB has the default p
		# eventually we should find a way to implement that functionality
		# could use normal distribution (some of normals is easy) or Poisson Binomial (sum of independent non-identical binomials)
		# for now use a dirty trick; after initial training, fix all CORE->N/V weights to 1
		# then continue "training", fixing the N/V assemblies after initial firing from PHON+CONTEXT
		self.CORE_k = CORE_k
		self.add_explicit_area(CORE, 2*CORE_k, self.CORE_k, 0.1, 
			custom_inner_p=0.9, custom_out_p=0.9, custom_in_p=0.9)
		self.update_plasticity(CORE, NOUN, 0.1)
		self.add_area(SEQ, LEX_n, LEX_k, beta)
		self.add_explicit_area(MOOD, 1*EXPLICIT_k, EXPLICIT_k, beta)

	def parse_with_syntax(self, sentence, mood_state=0):
		# SEQ always projects to both cores, has to project to a core when it is also activated
		# CORES are in mutual inhibition... one with more input fires
		# CAT JUMP
		# activate MOOD[INDICATIVE] -> SEQ
		# activate NOUN[CAT] -> NOUN_CORE (at same time SEQ -> CORES)
		# NOUN_CORE -> SEQ 
		# activate VERB[JUMP] -> VERB_CORE (at same time SEQ -> CORES)

		self.activate(MOOD, mood_state)
		self.project({}, {MOOD: [SEQ]})
		for _ in range(self.proj_rounds):
			self.project({}, {MOOD: [SEQ], SEQ: [SEQ]})

		# key idea: we only want cores interacting with SEQ (so SEQ <-> cores only)
		for word in sentence:
			self.activate_PHON(word)
			self.project({}, {PHON: [NOUN, VERB]})
			# mutual inhbition-- only one of NOUN/VERB fire next
			noun_input = self.get_input_from(PHON, NOUN)
			verb_input = self.get_input_from(PHON, VERB)
			area_firing_into_core = NOUN if (noun_input > verb_input) else VERB

			self.project({}, {SEQ: [CORE, SEQ], area_firing_into_core: [CORE]})
			# Could capture more complexity by projecting CORE,SEQ -> SEQ (can hold some state this way)
			self.project({}, {CORE: [SEQ]})
			for _ in range(self.proj_rounds):
				self.project({}, {CORE: [SEQ], SEQ: [SEQ]})

	def train_cores(self, rounds=20):
		# hack to get around common p value in all areas
		updated_weight = (1+self.area_by_name[CORE].area_beta[NOUN]) ** rounds

		core_to_noun_shape = ((self.connectomes[CORE][NOUN])[0:self.CORE_k,:]).shape
		(self.connectomes[CORE][NOUN])[0:self.CORE_k,:] = np.ones(core_to_noun_shape) * updated_weight

		noun_to_core_shape = ((self.connectomes[NOUN][CORE])[:, 0:self.CORE_k]).shape
		(self.connectomes[NOUN][CORE])[:, 0:self.CORE_k] = np.ones(noun_to_core_shape) * updated_weight

		core_to_verb_shape = ((self.connectomes[CORE][VERB])[self.CORE_k:, :]).shape
		(self.connectomes[CORE][VERB])[self.CORE_k:, :] = np.ones(core_to_verb_shape) * updated_weight

		verb_to_core_shape = ((self.connectomes[VERB][CORE])[:, self.CORE_k:]).shape
		(self.connectomes[VERB][CORE])[:, self.CORE_k:] = np.ones(verb_to_core_shape) * updated_weight

	def train_syntax(self, order, train_rounds=40):
		if order == "NV":
			sentence = ["DOG", "JUMP"]
		if order == "VN":
			sentence = ["JUMP", "DOG"]
		for _ in train_rounds:
			self.parse_with_syntax(sentence)

class LearnBrain_Syntax():
	def __init__(self, p, PHON_k=100, CONTEXTUAL_k=100, EXPLICIT_k=100, LEX_k=100, LEX_n=10000, beta=0.06, proj_rounds=2,
		CORE_k=10, bilingual=False, LANG_k=100, num_nouns=2, num_verbs=2, extra_context_areas=0, extra_context_area_k=10, extra_context_model="B", extra_context_delay=0):
		super().__init__(p, PHON_k=100, CONTEXTUAL_k=100, EXPLICIT_k=100, LEX_k=100, LEX_n=10000, beta=0.06, proj_rounds=2,
			CORE_k=10, bilingual=False, LANG_k=100, num_nouns=2, num_verbs=2, extra_context_areas=0, extra_context_area_k=10, extra_context_model="B", extra_context_delay=0)
		# note that custom_out_p only works into *other explicit areas*
		# that means CORE -> NOUN/VERB has the default p
		# eventually we should find a way to implement that functionality
		# could use normal distribution (some of normals is easy) or Poisson Binomial (sum of independent non-identical binomials)
		# for now use a dirty trick; after initial training, fix all CORE->N/V weights to 1
		# then continue "training", fixing the N/V assemblies after initial firing from PHON+CONTEXT
		self.CORE_k = CORE_k
		self.add_explicit_area(CORE, 2*CORE_k, self.CORE_k, 0.1, 
			custom_inner_p=0.9, custom_out_p=0.9, custom_in_p=0.9)
		self.update_plasticity(CORE, NOUN, 0.1)
		self.add_area(SEQ, LEX_n, LEX_k, beta)
		self.add_explicit_area(MOOD, 1*EXPLICIT_k, EXPLICIT_k, beta)

	# DELETE EVENTUALLY
	def OBJECTS_train_cores(self, rounds=20):
		# hack to get around common p value in all areas
		updated_weight = (1+self.area_by_name[CORE].area_beta[NOUN]) ** rounds

		core_to_noun_shape = ((self.connectomes[CORE][NOUN])[0:self.CORE_k,:]).shape
		(self.connectomes[CORE][NOUN])[0:self.CORE_k,:] = np.ones(core_to_noun_shape) * updated_weight

		noun_to_core_shape = ((self.connectomes[NOUN][CORE])[:, 0:self.CORE_k]).shape
		(self.connectomes[NOUN][CORE])[:, 0:self.CORE_k] = np.ones(noun_to_core_shape) * updated_weight

		self.disable_plasticity = True
		# 0 index word is an INTRANSITIVE verb
		# SECOND core is for intrans verb
		self.activate(MOTOR, 0)
		self.project({}, {MOTOR: [VERB]})
		intrans_verb_assembly = self.area_by_name[VERB].winners[:]
		for i in intrans_verb_assembly:
			(self.connectomes[CORE][VERB])[self.CORE_k:(2*self.CORE_k), i] = updated_weight
			(self.connectomes[VERB][CORE])[i, self.CORE_k:(2*self.CORE_k)] = updated_weight

		# 1 index word is a TRANSITIVE verb
		# THIRD core is for trans verb
		self.activate(MOTOR, 1)
		self.project({}, {MOTOR: [VERB]})
		trans_verb_assembly = self.area_by_name[VERB].winners[:]
		for i in trans_verb_assembly:
			(self.connectomes[CORE][VERB])[2*self.CORE_k:, i] = updated_weight
			(self.connectomes[VERB][CORE])[i, 2*self.CORE_k:] = updated_weight

		o = bu.overlap(intrans_verb_assembly, trans_verb_assembly, percentage=True)
		print("Got overlap of trans and intrans assemblies of " + str(o))

		self.disable_plasticity = False

	# DELETE EVENTUALLY
	def OBJECTS_train_syntax(train_rounds=20):
		# example language is S iV, S O tV

		# MOTOR[0] -> VERB -> CORES -> SEQ
		# whatever we get, fuse to ROLES[0] (which is for SUBJ) and to CORES[0] (noun)
		# SEQ + CORE -> SEQ
		# whatever we get, fuse to CORES[1] (intrans verb)
		self.activate(CORE, 1)
		self.activate(SEQ, 0)
		self.activate(ROLES, 0)
		self.disable_plasticity = False
		for _ in range(train_rounds):
			project({}, {CORE: [SEQ], SEQ: [ROLES]})
		self.activate(CORE, 0)
		for _ in range(train_rounds):
			project({}, {SEQ: [CORE]})

		SEQ_updated_weight = (1+self.area_by_name[SEQ].area_beta[SEQ]) ** train_rounds
		(self.connectomes[SEQ][SEQ])[:self.SEQ_k, self.SEQ_k:(2*self.SEQ_k)] = SEQ_updated_weight
		self.activate(SEQ, 1)
		self.activate(CORE, 1)
		for _ in range(train_rounds):
			project({}, {SEQ: [CORE]})

		# MOTOR[1] -> VERB -> CORES -> SEQ
		# whatever we get, fuse to ROLES[0] (which is for SUBJ) and to CORES[0]
		# SEQ + CORE -> SEQ
		# whatever we get, fuse to ROLES[1] (which is for OBJ) and to CORES[0] (noun)
		# SEQ + CORE -> SEQ 
		# whatever we get, fuse to CORES[2] (intrans verb)
		self.activate(CORE, 2)
		self.activate(SEQ, 2)
		self.activate(ROLES, 0)
		for _ in range(train_rounds):
			project({}, {CORE: [SEQ], SEQ: [ROLES]})
		self.activate(CORE, 0)
		for _ in range(train_rounds):
			project({}, {SEQ: [CORE]})

		(self.connectomes[SEQ][SEQ])[(2*self.SEQ_k):(3*self.SEQ_k), (3*self.SEQ_k):(4*self.SEQ_k)] = SEQ_updated_weight
		self.activate(SEQ, 3)
		self.activate(CORE, 0)
		self.activate(ROLES, 1) # represents object!!
		for _ in range(train_rounds):
			project({}, {CORE: [SEQ], SEQ: [ROLES, CORE]})

		(self.connectomes[SEQ][SEQ])[(3*self.SEQ_k):(4*self.SEQ_k), (4*self.SEQ_k):(5*self.SEQ_k)] = SEQ_updated_weight
		self.activate(SEQ, 4)
		self.activate(CORE, 0)
		for _ in range(train_rounds):
			project({}, {CORE: [SEQ]})
		self.activate(CORE, 2) # transitive verb!!
		for _ in range(train_rounds):
			project({}, {SEQ: [CORE]})
