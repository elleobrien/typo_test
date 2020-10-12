import nltk
import random

f = open("data/raw.txt")
corpus = f.read()

# Split into sentences
sent_list = nltk.tokenize.sent_tokenize(corpus)
# Remove any sentences that are suspiciously short - say <= 20 characters
clean_list = [s for s in sent_list if len(s) > 20]

# Randomly select 1000 for testing
random.seed(1)
keep = random.sample(clean_list, 500)

# Write this subset to file
with open('data/test_set.tsv', 'w') as f:
    for item in keep:
        # Remove any newlines in the body of the text to avoid confusion
        f.write("%s\t" % item.strip())
