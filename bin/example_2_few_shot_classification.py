# general imports
import sys
import os
import pandas as pd
import torch
torch.backends.cuda.matmul.allow_tf32 = True
path_to_package = os.path.split(os.path.split(__file__)[0])[0]
sys.path.append(os.path.join(path_to_package, 'src'))
from llm_model import LLM
from llm_classifier import LLMClassifier
import constants
import evaluation


# Get the data and prompt formatting for the task; we are going to be using sentiment analysis on Amazon reviews as an example
task_name = 'amazon_reviews'
n_test = 200  # number of test examples
n_in_context = 5  # number of in-context examples to give in the prompt
# Load the data
df = pd.read_csv(os.path.join(path_to_package, 'data', task_name, 'test.csv'), sep='\t')  # all data
# Extract a test set to run classification on and some labelled examples to provide in the prompt
df_test = df[:n_test]  # test split
samples_test = df_test[constants.TEXT].values  # text inputs
gt_labels_test = df_test[constants.GROUND_TRUTH_LABEL].values.astype(int)  # classes outputs as integers
df_in_context = df[n_test:n_test+n_in_context]  # in-context exmples
samples_in_context = df_in_context[constants.TEXT].values  # text inputs
gt_labels_in_context = df_in_context[constants.GROUND_TRUTH_LABEL].values.astype(int)  # classes outputs as integers
# Get the prompt formatting functions for this task (saved as a script in the dataset folder)
sys.path.append(os.path.join(path_to_package, 'data', task_name))
import prompts  # script with prompt formatting functions

# Define the back-bone LLM to use
# llm = LLM(model_name="mistralai/Mistral-7B-Instruct-v0.3", use_reduced_precision=True)
llm = LLM(model_name="google/gemma-7b-it", use_reduced_precision=True)

# define LLM-based classifier class
classifier = LLMClassifier(model=llm, prompt_formatting=prompts)

# let's print out an example of the prompt we will feed in to make sure it looks correct with the current settings
classifier.print_prompt_example(input_examples=samples_in_context, labels_examples=gt_labels_in_context)

# run on the test set
output_probs = classifier.soft_labels_batch(input_texts=samples_test, input_examples=samples_in_context, labels_examples=gt_labels_in_context)

# look at some output examples and ground-truth labels
print('Output probabilities:')
print(output_probs[:10, :])
print('Ground-truths:')
print(gt_labels_test[:10])

# evaluate output
f1_score = evaluation.compute_metric(gt_labels_test, output_probs, 'f1')
ece = evaluation.compute_metric(gt_labels_test, output_probs, 'ece')
print('f1-score: {}, ECE: {}'.format(f1_score, ece))