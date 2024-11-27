import json
import nltk
from nltk.tokenize import MWETokenizer

with open('data/lung_cancer_entities_batch_1.json', 'r') as file:
    data = json.load(file)

note_terms = {}    # KV pair of note_id : list of terms
non_med_terms_freq = {}  # KV pair of term : freq
attributes = ['pretty_name', 'source_value']    # meta_anns will be treated differntly since the value is a json object

# helper function to classify whether or not the given word is medical term
demographic_terms = {"year", "old"}
tokenizer = MWETokenizer()  #TODO: find other tokenizers better for general information purposes
def classify_term(term):
    # tokens = tokenizer.tokenize(term.lower())
    tokens = term.lower().split()
    # print(tokens)
    return True # @TODO: now returns everything to be true, can use negation of some medical corpus
    

c = 0
for note in data:
    terms = []  # to store terms temporarily
    note_id = note['note_id']
    entities = note['entities']['entities']
    for section_id in entities:
        for attr in attributes: # collect pretty_name and source_value in each note segment
            # print(entities[section_id][attr])
            terms.append(entities[section_id][attr])

# parse terms list, update non_med_results list using classifier
    note_terms.update({note_id: terms})
    for term in terms:
        if (classify_term(term)):
            if term in non_med_terms_freq:
                non_med_terms_freq.update({term : non_med_terms_freq.get(term) + 1})
            else:
                non_med_terms_freq.update({term : 1})


print(non_med_terms_freq)
