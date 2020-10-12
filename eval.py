from transformers import pipeline, AutoTokenizer
import nlpaug.augmenter.char as nac
import json

# Load model 
model = pipeline("sentiment-analysis",model="distilbert-base-uncased")

# Define text perturbation
aug = nac.KeyboardAug() # Insert realistic keystroke errors
def typo(input):
    output = aug.augment(input)
    return(output)

def eval_perturb(input_a,input_b):
    output_a, output_b = model([input_a, input_b])
    sq_error = (output_a["score"] - output_b["score"])**2
    acc = output_a["label"] == output_b["label"]
    return(sq_error,acc)


# Read in our test dataset
f = open("data/test_set.tsv")
test_dataset = f.read().split("\t")[:-1]

# Loop over all test examples and evaluate
mse, total_acc = 0,0
n = len(test_dataset)
for sentence in test_dataset:
    sq_error, acc = eval_perturb(sentence, typo(sentence))
    mse += (1/n) * sq_error
    total_acc += (1/n) * acc

# Write results to file
with open("perturbation_test.json", 'w') as outfile:
        json.dump({ "accuracy": total_acc, "mse":mse}, outfile)

