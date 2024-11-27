import nltk
from nltk.tokenize import MWETokenizer

tokenizer = MWETokenizer()

term = "i am a heavy smoker"

tokens = tokenizer.tokenize(term.lower())
print(tokens)