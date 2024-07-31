# general imports
import sys
import numpy as np
import os
import pandas as pd
import torch
torch.backends.cuda.matmul.allow_tf32 = True
path_to_package = os.path.split(os.path.split(__file__)[0])[0]
sys.path.append(os.path.join(path_to_package, 'src'))
from bpe import BayesPE
import constants
import evaluation


# Get the data and prompt formatting for the task; we are going to be using sentiment analysis on Amazon reviews as an example
task_name = 'amazon_reviews'
n_val = 100
n_test = 200
# Load the data
df = pd.read_csv(os.path.join(path_to_package, 'data', task_name, 'test.csv'), sep='\t')  # all data
# Extract a validation and a test sets
df_val = df[:n_val]  # validation split
df_test = df[n_val:n_val+n_test]  # test split
samples_val = df_val[constants.TEXT].values  # text inputs
gt_labels_val = df_val[constants.GROUND_TRUTH_LABEL].values.astype(int)  # classes outputs as integers
samples_test = df_test[constants.TEXT].values  # text inputs
gt_labels_test = df_test[constants.GROUND_TRUTH_LABEL].values.astype(int)  # classes outputs as integers

# Get the prompt formatting functions for this task
sys.path.append(os.path.join(path_to_package, 'data', task_name))
import prompts as prompt_formatting  # script with prompt formatting functions

# Now let's define a series of semantically equivalent prompt instructions we will use for BPE.
# You can define these in any way; manually, rephrasing an initial one with an LLM or using automated methods like APE.
instructions = [
'classify the sentiment of the Amazon review below into one of the following classes:',
'Categorize the sentiment of the Amazon review provided into one of the following classes:',
'Categorize the sentiment of the Amazon review provided into one of the given classes:',
'Determine the sentiment category of the given Amazon review by classifying it into one of the following classes:',
'Classify the sentiment of the given Amazon review into one of the following categories:',
'Assign the sentiment of the Amazon review provided to one of the given categories:',
'Categorize the sentiment of the provided Amazon review into one of the following classes:',
'Determine the sentiment category that best corresponds to the Amazon review provided amongst the following options:',
'Classify the sentiment expressed in the Amazon review below into one of the following categories:'
]

# make a set of in-context examples for each prompt above
n_in_context = 5  # number of in-context examples to use
for i in range(len(instructions)):
    df_in_context = df[n_val+n_test+i*n_in_context:n_val+n_test+(i+1)*n_in_context]  # in-context exmples
    samples_in_context_i = df_in_context[constants.TEXT].values  # text inputs
    gt_labels_in_context_i = df_in_context[constants.GROUND_TRUTH_LABEL].values.astype(int)  # classes outputs as integers
    if i==0:
        samples_in_context = np.expand_dims(samples_in_context_i, axis=1)
        gt_labels_in_context = np.expand_dims(gt_labels_in_context_i, axis=1)
    else:
        samples_in_context = np.concatenate((samples_in_context, np.expand_dims(samples_in_context_i, axis=1)), axis=1)
        gt_labels_in_context = np.concatenate((gt_labels_in_context, np.expand_dims(gt_labels_in_context_i, axis=1)), axis=1)

# Define the BayesPE class. We are using here the Mistral-7B-Instruct as back-bone, but all HuggingFace LLMs should be supported (see paper for list of tested ones)
bayespe_classifier = BayesPE(model_name="google/gemma-7b-it", prompt_formatting=prompt_formatting, instructions=instructions, few_shot_texts_sets=samples_in_context, few_shot_labels_sets=gt_labels_in_context, max_len_content=200, use_reduced_precision=True)

# let's print out an example of the prompt we will feed in to make sure it looks correct with the current settings
bayespe_classifier.print_prompt_example()

# train the weights of the BPE
bayespe_classifier.optimise_weights(samples_val, gt_labels_val)
print(bayespe_classifier.weights)

# save the trained weights
bayespe_classifier.save_weights()

# run on the test set
output_probs = bayespe_classifier.forward(samples_test, n_forward_passes=None)

# look at some output examples and ground-truth labels
print('Output probabilities:')
print(output_probs[:10, :])
print('Ground-truths:')
print(gt_labels_test[:10])

# evaluate output
f1_score = evaluation.compute_metric(gt_labels_test, output_probs, 'f1')
ece = evaluation.compute_metric(gt_labels_test, output_probs, 'ece')
print('f1-score: {}, ECE: {}'.format(f1_score, ece))