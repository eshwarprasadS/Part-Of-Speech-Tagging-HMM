import sys
import math
from collections import defaultdict
import json
import numpy as np

def load_params(param_file):
	with open(param_file) as f:
		params = json.load(f)
	transition_probs = params['TransitionProbabilities']
	emission_probs = params['EmissionProbabilities']

	return transition_probs, emission_probs

class HMM_test:
	def __init__(self, transition_probs, emission_probs, test_data, tag_set, vocab):
		self.test_data = test_data
		self.transition_probs = transition_probs
		self.emission_probs = emission_probs
		self.tag_set = tag_set
		self.vocab = vocab
		self.open_class_threshold = 0.08
		self.open_class_tags = set()
		# self.max_probability = defaultdict(lambda: defaultdict(float))
		# self.back_pointer = defaultdict(lambda: defaultdict(str))
	
	def calc_open_class_tags(self):
		# Defining open class tags
		self.open_class_threshold = self.open_class_threshold * len(self.vocab)
		for tag in self.tag_set:
			if len(self.emission_probs[tag]) > self.open_class_threshold:
				self.open_class_tags.add(tag)

	def write_to_output(self, file_path):
		hmmoutput = ""
		for line in self.test_data:
			hmmoutput += self.viterbi_decode_line(line) + "\n"

		# Save output to hmmoutput.txt
		with open(file=file_path, encoding='utf8', mode="w") as f:
			f.write(hmmoutput)

	# Viterbi Decoding

	def viterbi_decode_line(self, line):
		words = line.split()

		max_probability = defaultdict(lambda: defaultdict(float))
		back_pointer = defaultdict(lambda: defaultdict(str))

		# Initialize start tag
		for state in self.tag_set:
			# maintain only non-zero probability states. If theres zero probability, its because of unseen emission
			if words[0] in self.vocab and words[0] in self.emission_probs[state]:
				max_probability[0][state] = self.transition_probs["StArT"][state] + self.emission_probs[state][words[0]]
				back_pointer[0][state] = "StArT"
			elif words[0] not in self.vocab:
				max_probability[0][state] = self.transition_probs["StArT"][state]
				back_pointer[0][state] = "StArT"
			
		for index in range(1, len(words)):
		
			for state in self.tag_set:
				if words[index] in self.vocab and words[index] in self.emission_probs[state]: 
					max_probability[index][state], back_pointer[index][state] = self.get_max_probability(state, max_probability[index-1], self.transition_probs)
					max_probability[index][state] += self.emission_probs[state][words[index]] # Add emission log prob

				# If word is not found in vocabulary, then just consider transition probability
				elif words[index] not in self.vocab:
					# if word not in vocab, consider only open_class tags, if not, continue
					if state not in self.open_class_tags:
						continue
					else:
						max_probability[index][state], back_pointer[index][state] = self.get_max_probability(state, max_probability[index-1], self.transition_probs)

		final_tag = ""
		final_prob = -np.inf

		# Find the best state for final probability, equate that to final_tag.
		for state in max_probability[len(words)-1]:
			# consider probability from final tag to end tag
			if max_probability[len(words)-1][state] + self.transition_probs[state]["eNd"] > final_prob:
				final_tag = state
				final_prob = max_probability[len(words)-1][state] + self.transition_probs[state]["eNd"]

		# Backtrace path
		tagged_line = []
		predicted_tag = final_tag
		for i in range(len(words) - 1, -1, -1):
			tagged_line.append(str(words[i]) + '/' + str(predicted_tag))
			predicted_tag = back_pointer[i][predicted_tag]

		return " ".join(tagged_line[::-1])

# Function to find max and argmax 

	def get_max_probability(self, current_state, path_probability, transition_probs):
		back_pointer = ""
		max_prob = -np.inf

		for state in path_probability:

			if max_prob < path_probability[state]+transition_probs[state][current_state]:
				max_prob =  path_probability[state]+transition_probs[state][current_state]
				back_pointer = state

		return max_prob, back_pointer


if __name__ == '__main__':

	#Load test data from argument
	test_data_path = sys.argv[1]
	with open(test_data_path, encoding='utf8') as f:
		test_data = f.readlines()

	# Load model parameters from model file
	transition_probs, emission_probs = load_params('./hmmmodel.txt')

	# Getting all known tags as 'tag-set'
	tag_set  = set(list(transition_probs.keys()))

	# Remove start tag, since it was made for convenience
	tag_set.remove("StArT")

	# Get the vocabulary, list of all known words
	vocab = set([word for tag in list(emission_probs.keys()) for word in list(emission_probs[tag].keys())])
	
	hmmtest = HMM_test(transition_probs, emission_probs, test_data, tag_set, vocab)
	hmmtest.calc_open_class_tags()
	hmmtest.write_to_output("./hmmoutput.txt")


