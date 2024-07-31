
## Description

This package implements the method described and evaluated in the paper 
"Bayesian Prompt Ensembles: Model Uncertainty Estimation for Black-Box Large Language Models".
Bayesian Prompt Ensembles (BayesPE) is a method to combine multiple semantically equivalent 
prompts to obtain well-calibrated output probabilities with Large Language Models. 
The package includes tools to i) perform classification through prompting with LLMs and 
ii) use the BayesPE approach to ensemble multiple prompts, improving calibration performance.
Below you will find a comprehensive tutorial divided into four self-contained parts:
1. **Zero-Shot Classification with an LLM:** Use an LLM to perform classification through prompting.
2. **Few-Shot Classification with an LLM:** Use an LLM to perform classification through in-context learning, providing a few labelled examples in the prompt.
3. **BayesPE for Zero-Shot Classification:** Use BayesPE to ensemble different semantically equivalent prompts to perform classification with an LLM.
4. **BayesPE for Few-Shot Classification:** Use BayesPE to ensemble different semantically equivalent prompts and in-context examples to perform classification with an LLM.

If you use this package, please cite our paper: https://www.amazon.science/publications/bayesian-prompt-ensembles-model-uncertainty-estimation-for-black-box-large-language-models

```console
@article{tonolini2024bayesian,
  title={Bayesian prompt ensembles: Model uncertainty estimation for black-box large language models},
  author={Tonolini, Francesco and Massiah, Jordan and Aletras, Nikolaos and Kazai, Gabriella},
  journal={Association for Computational Linguistics}
  year={2024}
}
```

If you have questions or need help, don't hesitate to get in touch: tonolini@amazon.com

## Installation

Copy this package to where you need it, then do the following:
1) move to the package's directory
```console
cd BayesPE
```

2) create a Python environment
```console
conda create --name bayespe python=3.10
```

3) activate the environment
```console
source activate bayespe
```

4) install requirements
```console
pip install -r requirements.txt
```

5) install Huggingface CLI
```console
pip install -U "huggingface_hub[cli]"
```

6) login to Huggingface (for access to LLMs) and enter your token.
```console
huggingface-cli login
```



And you are good to go!

## Example 1: Zero-Shot Classification with an LLM

Here is a simple example of classifying text with an LLM
using the package.

#### Imports

General imports:

```python
import sys
import os
import pandas as pd
```
Add the src directory to the path:
```python
path_to_package = os.path.split(os.path.split(__file__)[0])[0]
sys.path.append(os.path.join(path_to_package, 'src'))
```
Import relevant classes and scripts from src:
```python
from llm_model import LLM  # class for LLM wrapper
from llm_classifier import LLMClassifier  # class for classifier using LLMs
import evaluation  # evaluation functions
```

#### Load Data

We will be using sentiment classification of Amazon reviews
for appliances, where reviews are to be classified as either
positive or negative:

```python
df = pd.read_csv('data/amazon_reviews/test.csv', sep='\t')  # pandas DataFrame containing text strings and integer labels
```

Let's take 200 examples to classify, including text inputs and
numeric ground truth labels to compare with after inference:
```python
n_test = 200
df_test = df[:n_test]  # test split
samples_test = df_test['text'].values  # text inputs
gt_labels_test = df_test['ground_truth_label'].values.astype(int)  # classes ground-truths as integers
```
#### LLM and Prompt Formatting

Now we can call the LLM wrapper class to load the LLM of choice from
Huggingface. In this example we will use "mistralai/Mistral-7B-Instruct-v0.3";
a 7b instruction fine-tuned model from Mistral AI:
```python
llm = LLM(model_name="mistralai/Mistral-7B-Instruct-v0.3", use_reduced_precision=True)
```
We have used the "use_reduced_precision=True" argument, which will load
the model at bfloat16 precision, reducing memory requirements and making
the model much faster to run. For better performance, but higher compute
and memory, you can set this parameter to "False" or leave it as default.


Now we need to make some formatting functions and wrapping text to construct our
prompts and look for the right words at the output. These are
specific to the task and can be defined in a class or a separate
script. This class/script must hve the following objects:
```python
class PromptFormatting(object):
    def __init__(self):
        
        # 1) an instruction sentence
        INSTRUCTION = 'classify the sentiment of the Amazon review below into one of the following classes:'
        
        # 2) The words identifying the classes. In this case
        # 0 = negative and 1 = positive.
        self.CLASSES = [
            'negative',
            'positive'
        ]
        
        # 3) The list of options that will be given to the LLM
        # in the prompt (classes words in a numbered list)
        self.CLASSES_TEXT = '''1. {}
2. {}'''.format(self.CLASSES[0], self.CLASSES[1])

    def format_instruction(self, instruction):
        # 4) function which, given the instruction sentence, 
        # will put it together with the options list
        prompt = '''{}
{}
'''.format(instruction, self.CLASSES_TEXT)
        return prompt
    
    def format_content(self, content):
        # 5) formatting the text to be classified with a header and
        # the prompt to answer with one of the options. In this
        # case, the inputs are reviews.
        prompt = '''review: {}
the review is '''.format(content)
        return prompt

prompt_formatting = PromptFormatting()
 ```
You can play around with the objects in the 
class above to construct your prompts differently.
You can use this general format for any task.

Now we initialise the LLM classifier, which we can use to infer class
probabilities leveraging the LLM for prompting:
```python
classifier = LLMClassifier(model=llm, prompt_formatting=prompt_formatting)
```

The LLMClassifier class has a function to print out what the prompts will
look like and make sure it all looks ok:
```python
classifier.print_prompt_example()
```
This will return:
```console
classify the sentiment of the Amazon review below into one of the following classes:
1. negative
2. positive

review: <TEXT_IN>
the review is <LABEL_OUT>
```

#### Classify Examples

Now that we have our prompts and our LLM ready, we can run classification
on our set of 200 examples. The function "soft_labels_batch" will run classification
using the LLM for all inputs in the list "input_texts" and return class probabilities:
```python
output_probs = classifier.soft_labels_batch(input_texts=samples_test)
```
"output_probs" is a 2D n_samples x n_classes array containing the predicted class probabilities.
We can have a look at a few examples:
```python
print(output_probs[:10, :])
```
This returns something similar to the following:
```console
[[9.97527377e-01 2.47262316e-03]
 [4.13993755e-08 9.99999959e-01]
 [3.05902227e-07 9.99999694e-01]
 [1.12535162e-07 9.99999887e-01]
 [9.93307149e-01 6.69285092e-03]
 [9.82013790e-01 1.79862100e-02]
 [9.97527377e-01 2.47262316e-03]
 [1.12535162e-07 9.99999887e-01]
 [9.82013790e-01 1.79862100e-02]
 [3.05902227e-07 9.99999694e-01]]
```
This output is an array of probability of each of the two classes (negative and positive)
for each input sample inferred by the LLM.

#### Evaluate

Now we can test performance, using the evaluation scripts. For example,
we can look at f1-score for classification performance and ECE for calibration:
```python
f1_score = evaluation.compute_metric(gt_labels_test, output_probs, metric='f1')
ece = evaluation.compute_metric(gt_labels_test, output_probs, metric='ece')
print('f1-score: {}, ECE: {}'.format(f1_score, ece))
```
This will return something similar to:
```console
f1-score: 0.8897243107769424, ECE: 0.08265417069196701
```

With the "compute_metric" function you can compute the following metrics:

| metric | returns |
|:--------|:---------|
| 'f1' | macro f1-score |
| 'acc' | classification accuracy |
| 'nll' | negative log-likelihood |
| 'auc' | ROC-AUC score |
| 'ece' | expected calibration error (ECE) |
| 'mce' | maximum calibration error (MCE) |
| 'brier' | Brier score |

## Example 2: Few-Shot Classification with an LLM

This example performs the same classification of example 1, but providing the LLM with some labelled
samples in the prompt. This strategy is referred to as few-shot classification or in-context learning.

#### Imports

General imports:

```python
import sys
import os
import pandas as pd
```
Add the src directory to the path:
```python
path_to_package = os.path.split(os.path.split(__file__)[0])[0]
sys.path.append(os.path.join(path_to_package, 'src'))
```
Import relevant classes and scripts from src:
```python
from llm_model import LLM  # class for LLM wrapper
from llm_classifier import LLMClassifier  # class for classifier using LLMs
import evaluation  # evaluation functions
```

#### Load Data

We will be using sentiment classification of Amazon reviews
for appliances, where reviews are to be classified as either
positive or negative:

```python
df = pd.read_csv('data/amazon_reviews/test.csv', sep='\t')  # pandas DataFrame containing text strings and integer labels
```

Let's take 200 examples to classify, including text inputs and
numeric ground truth labels to compare with after inference:
```python
n_test = 200
df_test = df[:n_test]  # test split
samples_test = df_test['text'].values  # text inputs
gt_labels_test = df_test['ground_truth_label'].values.astype(int)  # classes ground-truths as integers
```
We will also take 5 examples and associated labels to form a few-shot prompt, giving the LLM some examples
of the task we want it to perform:
```python
n_in_context = 5  # number of in-context examples to give in the prompt
df_in_context = df[n_test:n_test+n_in_context]  # in-context exmples
samples_in_context = df_in_context['text'].values  # text inputs
gt_labels_in_context = df_in_context['ground_truth_label'].values.astype(int)  # classes outputs as integers
```

#### LLM and Prompt Formatting

Now we can call the LLM wrapper class to load the LLM of choice from
Huggingface. In this example we will use "mistralai/Mistral-7B-Instruct-v0.3";
a 7b instruction fine-tuned model from Mistral AI:
```python
llm = LLM(model_name="mistralai/Mistral-7B-Instruct-v0.3", use_reduced_precision=True)
```
We have used the "use_reduced_precision=True" argument, which will load
the model at bfloat16 precision, reducing memory requirements and making
the model much faster to run. For better performance, but higher compute
and memory, you can set this parameter to "False" or leave it as default.


Now we need to make some formatting functions and wrapping text to construct our
prompts and look for the right words at the output. These are
specific to the task and can be defined in a class or a separate
script. This class/script must hve the following objects:
```python
class PromptFormatting(object):
    def __init__(self):
        
        # 1) an instruction sentence
        INSTRUCTION = 'classify the sentiment of the Amazon review below into one of the following classes:'
        
        # 2) The words identifying the classes. In this case
        # 0 = negative and 1 = positive.
        self.CLASSES = [
            'negative',
            'positive'
        ]
        
        # 3) The list of options that will be given to the LLM
        # in the prompt (classes words in a numbered list)
        self.CLASSES_TEXT = '''1. {}
2. {}'''.format(self.CLASSES[0], self.CLASSES[1])

    def format_instruction(self, instruction):
        # 4) function which, given the instruction sentence, 
        # will put it together with the options list
        prompt = '''{}
{}
'''.format(instruction, self.CLASSES_TEXT)
        return prompt
    
    def format_content(self, content):
        # 5) formatting the text to be classified with a header and
        # the prompt to answer with one of the options. In this
        # case, the inputs are reviews.
        prompt = '''review: {}
the review is '''.format(content)
        return prompt

prompt_formatting = PromptFormatting()
 ```
You can play around with the objects in the 
class above to construct your prompts differently.
You can use this general format for any task.

Now we initialise the LLM classifier, which we can use to infer class
probabilities leveraging the LLM for prompting:
```python
classifier = LLMClassifier(model=llm, prompt_formatting=prompt_formatting)
```

The LLMClassifier class has a function to print out what the prompts will
look like and make sure it all looks ok. We can call this function with the in-context
examples and labels as arguments to see the resulting prompt that is given to the LLM:
```python
classifier.print_prompt_example(input_examples=samples_in_context, labels_examples=gt_labels_in_context)
```
This will return:
```console
EXAMPLE 1:
classify the sentiment of the Amazon review below into one of the following classes:
1. negative
2. positive

review: Installed this in my fridge, resettled the light and still shines red. Water come so out just fine, just not sure if it's our fridge or the filter.
the review is negative

EXAMPLE 2:
classify the sentiment of the Amazon review below into one of the following classes:
1. negative
2. positive

review: It had a decent size dent in the door.
the review is negative

EXAMPLE 3:
classify the sentiment of the Amazon review below into one of the following classes:
1. negative
2. positive

review: Good
the review is positive

EXAMPLE 4:
classify the sentiment of the Amazon review below into one of the following classes:
1. negative
2. positive

review: This is a perfect replacement for our KitchenAid utensil rack that had several holes in the bottom.
the review is positive

EXAMPLE 5:
classify the sentiment of the Amazon review below into one of the following classes:
1. negative
2. positive

review: I ordered one before this and it worked as good as the original factory one.  I will continue to buy from this company
the review is positive

EXAMPLE 6:
classify the sentiment of the Amazon review below into one of the following classes:
1. negative
2. positive

review: <TEXT_IN>
the review is <LABEL_OUT>
```
The prompt above lists five examples where we have provided the correct answer. We then initiate a sixth
example, where we will input the test sample in <TEXT_IN> and let the LLM chose the class at <LABEL_OUT>.
This will be automatically applied to all test examples during inference (see below).

#### Classify Examples

Now that we have our prompts and our LLM ready, we can run classification
on our set of 200 examples. The function "soft_labels_batch" will run classification
using the LLM for all inputs in the list "input_texts", using provided in-context examples
and labels to construct the prompt. The output will be class probabilities:
```python
output_probs = classifier.soft_labels_batch(input_texts=samples_test, input_examples=samples_in_context, labels_examples=gt_labels_in_context)
```
"output_probs" is a 2D n_samples x n_classes array containing the predicted class probabilities.
We can have a look at a few examples:
```python
print(output_probs[:10, :])
```
This returns something similar to the following:
```console
[[9.99664650e-01 3.35350130e-04]
 [3.05902227e-07 9.99999694e-01]
 [8.31528028e-07 9.99999168e-01]
 [8.31528028e-07 9.99999168e-01]
 [9.99876605e-01 1.23394576e-04]
 [9.99088949e-01 9.11051194e-04]
 [9.99876605e-01 1.23394576e-04]
 [2.26032430e-06 9.99997740e-01]
 [9.99088949e-01 9.11051194e-04]
 [2.26032430e-06 9.99997740e-01]]
```
This output is an array of probability of each of the two classes (negative and positive)
for each input sample inferred by the LLM.

#### Evaluate

Now we can test performance, using the evaluation scripts. For example,
we can look at f1-score for classification performance and ECE for calibration:
```python
f1_score = evaluation.compute_metric(gt_labels_test, output_probs, metric='f1')
ece = evaluation.compute_metric(gt_labels_test, output_probs, metric='ece')
print('f1-score: {}, ECE: {}'.format(f1_score, ece))
```
This will return something similar to:
```console
f1-score: 0.934998374959374, ECE: 0.06773155927658081
```

## Example 3: BayesPE for Zero-Shot Classification

In this example we will show how to use BayesPE to combine multiple prompt instructions and
improve calibration of the resulting classification. BayesPE learns how "good" each
instruction is with a labelled validation set and weights them accordingly. At inference
time, we can set a budget of forward passes through the LLM to balance performance and 
cost. For example, setting the budget to 1 will simply choose the best performing prompt and
run classification with it.

#### Imports

General imports:

```python
import sys
import os
import pandas as pd
```
Add the src directory to the path:
```python
path_to_package = os.path.split(os.path.split(__file__)[0])[0]
sys.path.append(os.path.join(path_to_package, 'src'))
```
Import relevant classes and scripts from src:
```python
from bpe import BayesPE  # the BayesPE class
import evaluation  # evaluation functions
```

#### Load Data

We will be using sentiment classification of Amazon reviews
for appliances, where reviews are to be classified as either
positive or negative:

```python
df = pd.read_csv('data/amazon_reviews/test.csv', sep='\t')  # pandas DataFrame containing text strings and integer labels
```

We will take 100 examples for validation and 200 examples for testing.
Both will include text inputs and numeric ground truth labels. For the test set,
the ground-truth labels will be used for evaluation.
```python
# Validation set
n_val = 100
df_val = df[:n_val]  # validation split
samples_val = df_val['text'].values  # text inputs
gt_labels_val = df_val['ground_truth_label'].values.astype(int)  # classes outputs as integers
# Test set
n_test = 200
df_test = df[n_val:n_val+n_test]  # test split
samples_test = df_test['text'].values  # text inputs
gt_labels_test = df_test['ground_truth_label'].values.astype(int)  # classes outputs as integers
```

#### Prompt Formatting and Instructions

We need to make some formatting functions and wrapping text to construct our
prompts and look for the right words at the output. These are
specific to the task and can be defined in a class or a separate
script. This class/script must hve the following objects:
```python
class PromptFormatting(object):
    def __init__(self):
        
        # 1) an instruction sentence
        INSTRUCTION = 'classify the sentiment of the Amazon review below into one of the following classes:'
        
        # 2) The words identifying the classes. In this case
        # 0 = negative and 1 = positive.
        self.CLASSES = [
            'negative',
            'positive'
        ]
        
        # 3) The list of options that will be given to the LLM
        # in the prompt (classes words in a numbered list)
        self.CLASSES_TEXT = '''1. {}
2. {}'''.format(self.CLASSES[0], self.CLASSES[1])

    def format_instruction(self, instruction):
        # 4) function which, given the instruction sentence, 
        # will put it together with the options list
        prompt = '''{}
{}
'''.format(instruction, self.CLASSES_TEXT)
        return prompt
    
    def format_content(self, content):
        # 5) formatting the text to be classified with a header and
        # the prompt to answer with one of the options. In this
        # case, the inputs are reviews.
        prompt = '''review: {}
the review is '''.format(content)
        return prompt

prompt_formatting = PromptFormatting()
 ```
You can play around with the objects in the 
class above to construct your prompts differently.
You can use this general format for any task.

Next, we need to define the different prompt instructions we are going to ensemble with
BayesPE. These are semantically equivalent instructions for the task at hand, stored in a list of strings. In our paper,
We investigated many strategies to automatically generate these. In this tutorial we will manually 
define them. Let's make 9:
```python
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
 ```
Each of these will take the place of PromptFormatting.INSTRUCTIONS when iteratively running 
the LLM to form the ensemble.

#### Initialising and Optimising BayesPE

With the prompt formatting and our ensemble of instructions ready, we can initialise the BayesPE
classifier and optimise the ensemble weights with the validation set. First, we initialise
the BayesPE class:
```python
bayespe_classifier = BayesPE(model_name="mistralai/Mistral-7B-Instruct-v0.3", prompt_formatting=prompts, instructions=instructions, use_reduced_precision=True)
```
The BayesPE class takes as arguments the huggingface name of the underlying LLM to
be used (in this case Mistral-7b-Instruct), the prompt formatting class or script, the list of semantically equivalent 
instructions and, optionally, a boolean argument indicating whether to load the model
at reduced precision for efficiency (set to 'True' in this example). There are a few additional 
optional arguments (see doc string for details).

Similarly to the LLMClassifier class, the BayesPE class has a function to print out what
the prompts will look like and make sure it all looks ok:
```python
bayespe_classifier.print_prompt_example()
```
This will return the prompt that will be used for the LLM, using the first instruction
in the list:
```console
classify the sentiment of the Amazon review below into one of the following classes:
1. negative
2. positive

review: <SAMPLE_IN>
the review is <LABEL_OUT>
```
If the prompt looks ok, we can now run the LLM with all instructions on the validation set
and optimise the BayesPE prompts' weights. This is done by simply running the following
function:
```python
bayespe_classifier.optimise_weights(samples_val, gt_labels_val)
```
The above optimises the weights to assign to each instruction when running inference
using the validation samples and associated labels.

#### Inference with BayesPE

Now that the weights are optimised, we can use BayesPE to infer class probabilities for
test examples. We can decide our budget of LLM forward passes, up to the maximum available
instructions (in this case 9). BayesPE will start by using the most important instructions,
according to the optimised weights, and progressively work its way down. For example, if we set the forward
passes to 1, BayesPE will run once with the best instruction only. Let's try with 5:
```python
output_probs = bayespe_classifier.forward(samples_test, n_forward_passes=5)
```
"output_probs" is a 2D n_samples x n_classes array containing the predicted class probabilities.
We can have a look at a few examples:
```python
print(output_probs[:10, :])
```
This returns something similar to the following:
```console
[[7.32112607e-01 2.67887438e-01]
 [9.96170234e-01 3.82981073e-03]
 [1.01965173e-05 9.99989848e-01]
 [1.14533497e-05 9.99988591e-01]
 [1.39176421e-04 9.99860868e-01]
 [1.11139489e-05 9.99988931e-01]
 [8.41226263e-04 9.99158818e-01]
 [7.84371738e-01 2.15628307e-01]
 [1.15778909e-03 9.98842256e-01]
 [5.90006247e-05 9.99941044e-01]]
```
This output is an array of probability of each of the two classes (negative and positive)
for each input sample inferred by the LLM.

#### Evaluate

Now we can test performance, using the evaluation scripts. For example,
we can look at f1-score for classification performance and ECE for calibration:
```python
f1_score = evaluation.compute_metric(gt_labels_test, output_probs, metric='f1')
ece = evaluation.compute_metric(gt_labels_test, output_probs, metric='ece')
print('f1-score: {}, ECE: {}'.format(f1_score, ece))
```
This will return something similar to:
```console
f1-score: 0.8996386993175431, ECE: 0.07812481373548508
```


#### Save and Re-Load the BayesPE Weights

You can save the BayesPE weights after optimising them with the following function:
```python
bayespe_classifier.save_weights(save_dir='saved_weights/ensemble_weights')
```
This will save the weights as a Pickle object in the specified directory. 
Similarly, re-load weights saved in a given directory with:
```python
bayespe_classifier.load_weights(load_dir='saved_weights/ensemble_weights')
```

## Example 4: BayesPE for Few-Shot Classification

In this example we will show how to use BayesPE to combine multiple prompt instructions and
improve calibration of the resulting classification, similarly to example 3. However, we will
use BayesPE for in-context learning, providing the LLM with some labelled examples in the prompt.

#### Imports

General imports:

```python
import sys
import os
import pandas as pd
```
Add the src directory to the path:
```python
path_to_package = os.path.split(os.path.split(__file__)[0])[0]
sys.path.append(os.path.join(path_to_package, 'src'))
```
Import relevant classes and scripts from src:
```python
from bpe import BayesPE  # the BayesPE class
import evaluation  # evaluation functions
```

#### Load Data

We will be using sentiment classification of Amazon reviews
for appliances, where reviews are to be classified as either
positive or negative:

```python
df = pd.read_csv('data/amazon_reviews/test.csv', sep='\t')  # pandas DataFrame containing text strings and integer labels
```

We will take 100 examples for validation and 200 examples for testing.
Both will include text inputs and numeric ground truth labels. For the test set,
the ground-truth labels will be used for evaluation.
```python
# Validation set
n_val = 100
df_val = df[:n_val]  # validation split
samples_val = df_val['text'].values  # text inputs
gt_labels_val = df_val['ground_truth_label'].values.astype(int)  # classes outputs as integers
# Test set
n_test = 200
df_test = df[n_val:n_val+n_test]  # test split
samples_test = df_test['text'].values  # text inputs
gt_labels_test = df_test['ground_truth_label'].values.astype(int)  # classes outputs as integers
```

#### Prompts and In-Context Examples

We need to make some formatting functions and wrapping text to construct our
prompts and look for the right words at the output. These are
specific to the task and can be defined in a class or a separate
script. This class/script must hve the following objects:
```python
class PromptFormatting(object):
    def __init__(self):
        
        # 1) an instruction sentence
        INSTRUCTION = 'classify the sentiment of the Amazon review below into one of the following classes:'
        
        # 2) The words identifying the classes. In this case
        # 0 = negative and 1 = positive.
        self.CLASSES = [
            'negative',
            'positive'
        ]
        
        # 3) The list of options that will be given to the LLM
        # in the prompt (classes words in a numbered list)
        self.CLASSES_TEXT = '''1. {}
2. {}'''.format(self.CLASSES[0], self.CLASSES[1])

    def format_instruction(self, instruction):
        # 4) function which, given the instruction sentence, 
        # will put it together with the options list
        prompt = '''{}
{}
'''.format(instruction, self.CLASSES_TEXT)
        return prompt
    
    def format_content(self, content):
        # 5) formatting the text to be classified with a header and
        # the prompt to answer with one of the options. In this
        # case, the inputs are reviews.
        prompt = '''review: {}
the review is '''.format(content)
        return prompt

prompt_formatting = PromptFormatting()
 ```
You can play around with the objects in the 
class above to construct your prompts differently.
You can use this general format for any task.

Next, we need to define the different prompt instructions we are going to ensemble with
BayesPE. These are semantically equivalent instructions for the task at hand, stored in a list of strings. In our paper,
We investigated many strategies to automatically generate these. In this tutorial we will manually 
define them. Let's make 9:
```python
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
 ```
Each of these will take the place of PromptFormatting.INSTRUCTIONS when iteratively running 
the LLM to form the ensemble.

As we are performing classification with in-context learning, each instruction will need a
set of labelled examples to provide to the LLM. These can be defined for each instruction in
different ways. In this tutorial, we are simply going to use different random examples for
each instruction. We will take 5 examples for each instruction:
```python
n_in_context = 5  # number of in-context examples to use
for i in range(len(instructions)):  # for each instruction in the instructions list
    df_in_context = df[n_val+n_test+i*n_in_context:n_val+n_test+(i+1)*n_in_context]  # take 5 in-context exmples
    samples_in_context_i = df_in_context[constants.TEXT].values  # 5 text inputs
    gt_labels_in_context_i = df_in_context[constants.GROUND_TRUTH_LABEL].values.astype(int)  # 5 classes outputs as integers
    
    # concatenate over the iterations to form 2D arrays of input texts and labels
    if i==0:
        samples_in_context = np.expand_dims(samples_in_context_i, axis=1)
        gt_labels_in_context = np.expand_dims(gt_labels_in_context_i, axis=1)
    else:
        samples_in_context = np.concatenate((samples_in_context, np.expand_dims(samples_in_context_i, axis=1)), axis=1)
        gt_labels_in_context = np.concatenate((gt_labels_in_context, np.expand_dims(gt_labels_in_context_i, axis=1)), axis=1)
 ```
The result of the above are two 2D arrays, one of strings containing input texts and 
one of integers containing class labels, each of size n_in_context x n_instructions.
This is the format in which the BayesPE accepts in-context examples.

#### Initialising and Optimising BayesPE

With the prompt formatting and our ensemble of instructions ready, we can initialise the BayesPE
classifier and optimise the ensemble weights with the validation set. First, we initialise
the BayesPE class:
```python
bayespe_classifier = BayesPE(model_name="mistralai/Mistral-7B-Instruct-v0.3", prompt_formatting=prompt_formatting, instructions=instructions, few_shot_texts_sets=samples_in_context, few_shot_labels_sets=gt_labels_in_context, use_reduced_precision=True)
```
The BayesPE class takes as arguments the huggingface name of the underlying LLM to
be used (in this case Mistral-7b-Instruct), the prompt formatting class or script and the list of semantically equivalent 
instructions. As we are performing in-context learning, we have also provided the 2D arrays 
'few_shot_texts_sets' and 'few_shot_labels_sets', containing sets of text inputs and labels respectively
for each instruction in the ensemble. Optionally, we can define a boolean argument indicating whether to load the model
at reduced precision for efficiency (set to 'True' in this example). There are a few additional 
optional arguments (see doc string for details).

Similarly to the LLMClassifier class, the BayesPE class has a function to print out what
the prompts will look like and make sure it all looks ok:
```python
bayespe_classifier.print_prompt_example()
```
This will return an example of the prompt that will be given to the LLM, using the first instruction
in the list and the first set of in-context examples:
```console
EXAMPLE 1:
classify the sentiment of the Amazon review below into one of the following classes:
1. negative
2. positive

review: This is a mixed review. .. When I got the ice maker I was in love. I LOVE ice.. and it was making ice like a champ for about one month then slowly it started making half the cubes.. then 4 cubes.. then very thin see through cubes... to none. I will however say that the company has been very receptive to my returning it to be repaired. .. returning is always a pain in the butt and it seems so that a brandy new product should not be having any problems. Will letchu know how the "repair" turns out.
the review is negative

EXAMPLE 2:
classify the sentiment of the Amazon review below into one of the following classes:
1. negative
2. positive

review: I bought this in Feb 2016, so I have used it for a good 14 months now. The first problem is the oven does not always stay on after lighting. This is very irritating when you when you "think" you are pre-heating the oven and it is actually not on! Secondly, there is only one high temp burner, so forget about cooking a pot of water for pasta AND something else at the same time. Thirdly, the knobs are very cheap and easily moved so if you set an oven temperature and bump into the knob, it may no longer be set at the desired temperature. Finally, one of the burner knobs just broke. So....good luck if you buy this oven and expect to cook!
the review is negative

EXAMPLE 3:
classify the sentiment of the Amazon review below into one of the following classes:
1. negative
2. positive

review: I got what I thought was a great deal.  It was only used a couple of months and the people "remodeled" so they upgraded to a larger unit. Yeah.  First the doors just don't like to be shut.  That's why GE put a buzzer on it.  Second the drain gets plugged and it is a bear to remove the freezer drawers and the interior freezer back panel to clean it out.  Why hide it behind a panel?  It's noisy, has cheap refrigerator drawers.  The only thing good is it looks nice.

GE lost me as a customer for life after this.
the review is negative

EXAMPLE 4:
classify the sentiment of the Amazon review below into one of the following classes:
1. negative
2. positive

review: Ice maker did not work. Just kept leaking water all over the floor. leaked an entire 5 gallon jug in just a few hours.
the review is negative

EXAMPLE 5:
classify the sentiment of the Amazon review below into one of the following classes:
1. negative
2. positive

review: The packaging is very different from the one I bought from Home Depot.
the review is negative

EXAMPLE 6:
classify the sentiment of the Amazon review below into one of the following classes:
1. negative
2. positive

review: <SAMPLE_IN>
the review is <LABEL_OUT>
```
If the prompt looks ok, we can now run the LLM with all instructions on the validation set
and optimise the BayesPE prompts' weights. This is done by simply running the following
function:
```python
bayespe_classifier.optimise_weights(samples_val, gt_labels_val)
```
The above optimises the weights to assign to each instruction when running inference
using the validation samples and associated labels.

#### Inference with BayesPE

Now that the weights are optimised, we can use BayesPE to infer class probabilities for
test examples. We can decide our budget of LLM forward passes, up to the maximum available
instructions (in this case 9). BayesPE will start by using the most important instructions,
according to the optimised weights, and progressively work its way down. For example, if we set the forward
passes to 1, BayesPE will run once with the best instruction only. Let's try with 5:
```python
output_probs = bayespe_classifier.forward(samples_test, n_forward_passes=5)
```
"output_probs" is a 2D n_samples x n_classes array containing the predicted class probabilities.
We can have a look at a few examples:
```python
print(output_probs[:10, :])
```
This returns something similar to the following:
```console
[[9.55111911e-01 4.48880816e-02]
 [9.99915070e-01 8.49220944e-05]
 [1.29251932e-05 9.99987067e-01]
 [5.33277146e-05 9.99946665e-01]
 [3.05689604e-05 9.99969424e-01]
 [3.26815948e-05 9.99967311e-01]
 [1.08215687e-05 9.99989171e-01]
 [9.18138780e-01 8.18612125e-02]
 [1.31488799e-01 8.68511194e-01]
 [1.51307649e-05 9.99984862e-01]]
```
This output is an array of probability of each of the two classes (negative and positive)
for each input sample inferred by the LLM.

#### Evaluate

Now we can test performance, using the evaluation scripts. For example,
we can look at f1-score for classification performance and ECE for calibration:
```python
f1_score = evaluation.compute_metric(gt_labels_test, output_probs, metric='f1')
ece = evaluation.compute_metric(gt_labels_test, output_probs, metric='ece')
print('f1-score: {}, ECE: {}'.format(f1_score, ece))
```
This will return something similar to:
```console
f1-score: 0.9368717948717948, ECE: 0.04805548116564751
```

#### Save and Re-Load the BayesPE Weights

You can save the BayesPE weights after optimising them with the following function:
```python
bayespe_classifier.save_weights(save_dir='saved_weights/ensemble_weights')
```
This will save the weights as a Pickle object in the specified directory. 
Similarly, re-load weights saved in a given directory with:
```python
bayespe_classifier.load_weights(load_dir='saved_weights/ensemble_weights')
```
