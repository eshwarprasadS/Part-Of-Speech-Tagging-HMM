import sys
import math
from collections import defaultdict
import json
import numpy as np

class Train_HMM:
    def __init__(self, train_data):
        self.train_data = train_data
        self.emission_probabilities = defaultdict(lambda: defaultdict(float)) 
        self.transition_probabilities = defaultdict(lambda: defaultdict(float)) 
        self.tags = list
        self.parameters = {}
    def train(self):
        for data in self.train_data:
            tokens = data.split() #Each token is a "WORD/TAG" pair
            prev_tag = 'StArT' #assign an artificial start tag, make it in leetspeak so it wont collide with other meaningful 'start' tags that may appear

            for token in tokens:
                word, tag = token.rsplit("/", 1) # splitting on the final '/' as prof. Ron instructed to be careful with slashes
                #calculate transition frequences for tag to tag
                self.transition_probabilities[prev_tag][tag] = self.transition_probabilities[prev_tag][tag] + 1
                #calculate emission frequences for tag-word
                self.emission_probabilities[tag][word] = self.emission_probabilities[tag][word] + 1

                #update the prev tag to current tag
                prev_tag = tag
            
            #Introduce an 'end' tag which will never be a 'key' in my transition dictionary.
            self.transition_probabilities[prev_tag]['eNd'] = self.transition_probabilities[prev_tag]['eNd'] + 1

        #make a total tag list
        tags = list(self.transition_probabilities.keys())
        tags.append('eNd')
        self.tags = tags

        #Build log probabilties for transitions
        for prev_tag in self.transition_probabilities:
            
            # smoothing to handle unseen transitions by adding-1 uniformly
            prev_tag_denominator = 0
            for tag in self.tags:
                self.transition_probabilities[prev_tag][tag] += 1

                prev_tag_denominator += self.transition_probabilities[prev_tag][tag]
            
            #calculate the log probabilities for transitions
            for tag in self.transition_probabilities[prev_tag]:
                self.transition_probabilities[prev_tag][tag] = math.log(self.transition_probabilities[prev_tag][tag]/prev_tag_denominator)

        #Build log probabilties for transitions (no smoothing here, as per prof ron's insight)
        for tag in self.emission_probabilities:

            tag_denominator = 0
            for word in self.emission_probabilities[tag]:
                tag_denominator += self.emission_probabilities[tag][word]
            
            for word in self.emission_probabilities[tag]:
                self.emission_probabilities[tag][word] = math.log(self.emission_probabilities[tag][word]/tag_denominator)   

    def save_params(self, filepath):
        parameters = {'TransitionProbabilities' : self.transition_probabilities,
                        'EmissionProbabilities' : self.emission_probabilities}
        
        self.parameters = parameters

        with open(filepath, 'w', encoding = 'utf8') as f:
            f.write(json.dumps(self.parameters, indent = 4))

if __name__ == '__main__':

    train_data_path = sys.argv[1]
    with open(train_data_path, encoding='utf8') as f:
        train_data = f.readlines()

    hmmtrain = Train_HMM(train_data)
    hmmtrain.train()
    hmmtrain.save_params('./hmmmodel.txt')

