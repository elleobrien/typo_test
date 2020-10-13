from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import nlpaug.augmenter.char as nac
import json
import pandas as pd

# Load model 
tokenizer = AutoTokenizer.from_pretrained("textattack/albert-base-v2-SST-2")
inference_model = AutoModelForSequenceClassification.from_pretrained("textattack/albert-base-v2-SST-2")
model = pipeline("sentiment-analysis", model = inference_model,tokenizer=tokenizer)

# Define text perturbation
aug = nac.KeyboardAug(aug_word_max=1) # Insert realistic keystroke errors
def typo(input):
    output = aug.augment(input)
    return(output)

def eval_perturb(input_a,input_b):
    output_a, output_b = model([input_a, input_b])
    sq_error = (output_a["score"] - output_b["score"])**2
    acc = output_a["label"] == output_b["label"]
    return(sq_error,acc,output_b["score"])


# Read in our test dataset
f = open("data/test_set.tsv")
test_dataset = f.read().split("\t")[:-1]

# Loop over all test examples and evaluate
mse, total_acc = 0,0
n = len(test_dataset)
interesting_cases = []
for sentence in test_dataset:
    sentence_mod = typo(sentence)
    sq_error, acc, perturb_score = eval_perturb(sentence, sentence_mod)
    mse += (1/n) * sq_error
    total_acc += (1/n) * acc
    if acc == False:
        interesting_cases.append((sentence,sentence_mod,perturb_score))

interesting_cases.sort(key=lambda tup:tup[2], reverse=True)

# Write out our favorite interesting cases
to_report = interesting_cases[:5]
df = pd.DataFrame(to_report, columns = ["Original","Perturbed","Model confidence"])
with open("failure_modes.txt","w") as outfile:
    outfile.write(df.to_markdown(index=False))
    
# Write results to file
with open("test_score.json", 'w') as outfile:
        json.dump({ "accuracy": total_acc, "mse":mse}, outfile)

