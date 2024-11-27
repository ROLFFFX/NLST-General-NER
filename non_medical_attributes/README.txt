The focus of this part of the work is for General Name Entity Recognition, which is to identify 
NON-MEDICAL attributes (such as demographic information) that is also important to report.

Wordflow Explanation:

1. Data preparation
    - We have lung_cancer_entities_batch_x.json which is the structured clinical data, and 
    lung_cancer_notes.csv which map each data object with the report note via note_id.
    - Merge Data? 
    - Normalization such as all lowercase

2. Identify potential non-meical words
    - For each data object in the json file, identify potential non-medical attributes that is 
    worth collecting.
    - Cross compare with notes to validate and align the importance.

3. Attributes to parse:
    - pretty_name
    - source_value
    - meta_anns

4. Rank Importance:
    - when parsing each data entry, collect the non-medical keyword and store in some category 
    specific data structure. Then, rank each importance using some heuristics

--------

Regarding "Importance" heuristics
    - frequency analysis is a good place to start

Regarding Data Parsing and Temporary Data Storage

--------

Pseudocode for workflow:

dataStructure = {}

for each entry in data:
    note_id = entry.id  # use to link to lung_cancer_notes.csv
    note = lung_cancer_notes[note_id]
    terms = []
    for attribute in [pretty_name, source_value, meta_anns]:
        # for each attribute in entry
        terms.add(entry.attribute)  # add existing attribute content to terms
    
    # classify med or not
    non_med_attr = []
    for term in terms:
        if NLTK.classify(term) == "non_medical":  # Use NLP or predefined lexicon
            non_med_attr.add(term)

    # analyze importance of word, and store in dataStructure
    importance_scores = {}
    for term in non_med_attr:
        # Frequency analysis
        term_frequency = note.count(term)
        
        # Advanced heuristics (e.g., severity relevance, context weight)
        relevance_score = compute_relevance(term, severity_mapping)
        context_score = compute_context_weight(term, note)
        alignment_score = compute_alignment(term, key_indicators)
        
        # Combine scores using the composite formula
        importance_scores[term] = (
            frequency_weight * term_frequency +
            relevance_weight * relevance_score +
            context_weight * context_score +
            alignment_weight * alignment_score
        )

    # store to data_structure
    
    data_structure[note_id] = {
        "non_medical_terms": non_med_attr,
        "importance_scores": importance_scores
    }


