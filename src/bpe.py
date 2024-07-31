import numpy as np
import pickle
from llm_model import LLM
from llm_classifier import LLMClassifier
from ensemble_scaler import EnsembleScaler

SMALL_CONSTANT = 0.00001
LARGE_CONSTANT = 10000


def replace_nans_with_uniform(probs):
    """
    Function to replace probability distributions containing NaNs with uniform distributions (e.g. [NaN, 0.1] -> [0.5, 0.5])

    :param probs: 2D array of probability distributions [n_samples, n_classes]

    :return: 2D array of probability distributions with rows containing NaNs substituted with uniform [n_samples, n_classes]
    """
    for i in range(np.shape(probs)[0]):
        if np.isnan(probs[i,:]).any():
            probs[i, :] = np.divide(1.0, np.shape(probs)[1])*np.ones(np.shape(probs)[1])
    return probs


def smooth_probs_3d(probs_3d):
    """
    Function to format and smooth probability 3D arrays of probability distributions to avoid numerical precision errors.

    :param probs_3d: 3D array of probability distributions [n_samples, n_classes, n_instructions]

    :return: 3D array of probability distributions with rows containing NaNs substituted with uniform and smoothed out to avoid zeros [n_samples, n_classes, n_instructions]
    """
    probs_new = probs_3d + SMALL_CONSTANT
    for i in range(np.shape(probs_new)[2]):
        probs_new[:,:,i] = replace_nans_with_uniform(probs_new[:,:,i])
        for j in range(np.shape(probs_new)[0]):
            probs_new[j,:,i] = np.divide(probs_new[j,:,i], np.sum(probs_new[j,:,i]))
    return probs_new


class BayesPE(object):
    """
    Class for Bayesian Prompts Ensembles (BPE)
    """
    def __init__(self, model_name, prompt_formatting, instructions, few_shot_texts_sets=None, few_shot_labels_sets=None, max_len_content=None, use_reduced_precision=False, n_iterations_weights_optimiser=10):
        """
        :param model_name: the Huggingface name of the LLM to use (e.g. 'mistralai/Mistral-7B-Instruct-v0.1')
        :param prompt_formatting: formatting script to retrieve classes names and schemas from
        :param instructions: list or 1D array of strings with the different task instructions to construct the BPE
        :param few_shot_texts_sets: lists of text examples to include in the prompt for few-shot operation.
               Should be a list of lists where each list contains the set of examples to include in the prompt for
               each of the instructions in 'instructions'.
        :param few_shot_labels_sets: 2D array with the labels corresponding to the text examples in 'few_shot_texts_sets'.
               size should be [n_few_shot_examples, n_instructions].
        :param max_len_content: maximum word count for content. content texts longer than this will be truncated. In None, no truncation is applied
        :param use_reduced_precision: whether to use reduced precision for the LLM to use less GPU memory and compute
        :param n_iterations_weights_optimiser: number of iterations for the weights optimiser
        """
        model = LLM(model_name=model_name, use_reduced_precision=use_reduced_precision)
        self.classifier = LLMClassifier(model, prompt_formatting, max_len_content=max_len_content)
        self.scaler = EnsembleScaler(n_iterations_weights_optimiser)
        self.instructions = instructions
        if few_shot_texts_sets is not None:
            self.examples_dict = self.make_few_shot_examples_dict(few_shot_texts_sets, few_shot_labels_sets)
        else:
            self.examples_dict = None
        self.weights = np.divide(1.0, len(instructions))*np.ones(len(instructions))

    def optimise_weights(self, input_texts, gt_labels, learning_rate=SMALL_CONSTANT):
        """
        :param input_texts: list or 1D array of strings with the validation text input examples
        :param gt_labels: list or 1D array of ground-truth labels corresponding to input_texts
        :param learning_rate: initial learning rate for the weights optimiser

        :return: 1D array of optimised weights (w^* in the paper) [n_instructions].
        """
        probs = self.classifier.sample_probs_ensemble(self.instructions, input_texts, examples_dict=self.examples_dict, n_samples=len(self.instructions))
        probs = smooth_probs_3d(probs)
        nan_cost = True
        lr = learning_rate
        while nan_cost:
            optimal_weights, costs = self.scaler.train(probs, gt_labels, lr=lr)
            if not np.isnan(costs[-1]):
                nan_cost = False
            else:
                lr = lr * 0.5
        self.weights = optimal_weights
        return optimal_weights

    def forward(self, input_texts, n_forward_passes=None):
        if n_forward_passes is None:
            n_forward_passes = len(self.instructions)
        chosen_indices = np.argsort(self.weights)[-n_forward_passes:]
        chosen_weights = np.sort(self.weights)[-n_forward_passes:]
        probs = self.classifier.sample_probs_ensemble(self.instructions, input_texts, examples_dict=self.examples_dict, indices=chosen_indices)
        probs = smooth_probs_3d(probs)
        return self.scaler.scale_ensemble(probs, chosen_weights)

    def save_weights(self, save_dir='saved_weights/ensemble_weights'):
        with open(save_dir, 'wb') as f:
            pickle.dump(self.weights, f)

    def load_weights(self, load_dir='saved_weights/ensemble_weights'):
        with open(load_dir, 'rb') as f:
            self.weights = pickle.load(f)

    @staticmethod
    def make_few_shot_examples_dict(few_shot_texts_sets, few_shot_labels_sets):
        """
        :param few_shot_texts_sets: lists of text examples to include in the prompt for few-shot operation.
               Should be a list of lists where each list contains the set of examples to include in the prompt for
               each of the instructions in 'instructions'.
        :param few_shot_labels_sets: 2D array with the labels corresponding to the text examples in 'few_shot_texts_sets'.
               size should be [n_few_shot_examples, n_instructions].
        """
        examples_dict = {}
        for i in range(np.shape(few_shot_labels_sets)[1]):
            input_examples = few_shot_texts_sets[:, i]
            labels_examples = few_shot_labels_sets[:,i]
            examples_dict['input_examples_{}'.format(i)] = input_examples
            examples_dict['label_examples_{}'.format(i)] = labels_examples
        return examples_dict

    def print_prompt_example(self, index=0, input_text='<SAMPLE_IN>'):
        """
        Print out an example of the prompt that will be fed to the LLM with the current configuration.

        :param index: which prompt in the ensemble to print out an example for
        :param input_text: string with the input text to be evaluated.
        """
        if self.examples_dict is None:
            input_examples = None
            labels_examples = None
        else:
            input_examples = self.examples_dict['input_examples_{}'.format(index)]
            labels_examples = self.examples_dict['label_examples_{}'.format(index)]

        self.classifier.print_prompt_example(self.instructions[index], input_text, input_examples=input_examples, labels_examples=labels_examples)



