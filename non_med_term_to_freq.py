import json
import nltk
from nltk.tokenize import MWETokenizer 
from nltk.tokenize import TreebankWordTokenizer


with open('data/lung_cancer_entities_batch_1.json', 'r') as file:
    data = json.load(file)

non_med_results = {}    # KV pair of note_id : list of terms
attributes = ['pretty_name', 'source_value']    # meta_anns will be treated differntly since the value is a json object

# helper function to classify whether or not the given word is medical term
medical_terms = {"thoracic", "chest", "radiology", "cardiac", "diagnosis"}
tokenizer = MWETokenizer()
def classify_term(term):
    tokens = tokenizer.tokenize(term.lower())
    print(tokens)

c = 0
for note in data:
    terms = []  # to store terms temporarily
    note_id = note['note_id']
    entities = note['entities']['entities']
    for section_id in entities:
        for attr in attributes: # collect pretty_name and source_value in each note segment
            # print(entities[section_id][attr])
            terms.append(entities[section_id][attr])
    non_med_results.update({note_id: terms})

# print(non_med_results)

classify_term("The patient is a 85 year old heavy smoker, and he smokes a lot.")