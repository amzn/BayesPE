import numpy as np
from tqdm import tqdm

SMALL_CONSTANT = 0.00001
LARGE_CONSTANT = 10000


class LLMClassifier(object):
    """
    Class for classifier using an LLM
    """
    def __init__(self, model, prompt_formatting, max_len_content=None):
        """
        :param model: The LLM (class) to use for zero-shot classification
        :param prompt_formatting: formatting script or class to retrieve classes names and schemas from
        :param max_len_content: maximum word count for content. content texts longer than this will be truncated
        """
        self.model = model
        self.classes_strings = prompt_formatting.CLASSES
        self.instruction = prompt_formatting.INSTRUCTION
        self.classes_for_matching = prompt_formatting.CLASSES_FOR_MATCHING
        self.format_instruction = prompt_formatting.format_instruction
        self.format_content = prompt_formatting.format_content
        self.max_len_content = max_len_content

    def make_few_shot_instruction(self, instruction, input_examples=None, labels_examples=None):
        """
        Function to construct few-shot prompts, given an instruction and examples texts and labels.

        :param instruction: string containing the task instruction
        :param input_examples: list of strings containing the text of the examples to be used for few shot [n_dfew_shot_examples]
        :param labels_examples: list or 1D aray of int containing the ground-truth labels for the examples to be used for few shot [n_dfew_shot_examples]

        :return: string containing the few shot prompt to feed into the LLM
        """
        instruction = self.format_instruction(instruction)
        few_shots_instructions = ''
        for i in range(len(labels_examples)):
            class_i = self.classes_strings[labels_examples[i]]
            input_i = self.format_content(input_examples[i])
            input_i = self.truncate_text(input_i)
            text_i = '''EXAMPLE {}:
{}
{}{}

'''.format(i+1, instruction, input_i, class_i)
            few_shots_instructions = few_shots_instructions + text_i
        last_text = '''EXAMPLE {}:
{}'''.format(len(labels_examples) + 1, instruction)
        few_shots_instructions = few_shots_instructions + last_text
        return few_shots_instructions

    def truncate_text(self, input_text):
        """
        Function to truncate text given the max length in number of words (defined in self.max_len_content).

        :param input_text: string containing the text to be truncated

        :return: string containing the truncated text
        """
        input_text = str(input_text)
        if self.max_len_content is not None:
            if len(input_text.split())>self.max_len_content:
                output_text = ' '.join(input_text.split()[:self.max_len_content])
            else:
                output_text = input_text
        else:
            output_text = input_text
        return output_text

    def print_prompt_example(self, instruction=None, input_text='<TEXT_IN>', input_examples=None, labels_examples=None):
        """
        Print out the prompt fed to the LLM for inference.

        :param instruction: string containing the task instruction
        :param input_text: string containing the sample to be labelled
        """
        if instruction is None:
            instruction = self.instruction
        input_text = self.truncate_text(input_text)
        input_text = self.format_content(input_text)
        if input_examples is None or labels_examples is None:
            instruction = self.format_instruction(instruction)
        else:
            instruction = self.make_few_shot_instruction(instruction, input_examples=input_examples, labels_examples=labels_examples)
        prompt_text = '''{}
{}<LABEL_OUT>'''.format(instruction, input_text)
        print(prompt_text)

    def soft_label(self,  instruction=None, input_text='', input_examples=None, labels_examples=None):
        """
        Given task instructions and a sample text to be labelled, perform zero-shot classification with the LLM and returns label in class probabilities format.

        :param instruction: string containing the task instruction
        :param input_text: string containing the sample to be labelled

        :return: output label in class probabilities format as a 1D array [n_classes].
        """
        if instruction is None:
            instruction = self.instruction
        input_text = self.truncate_text(input_text)
        if input_examples is None or labels_examples is None:
            instruction = self.format_instruction(instruction)
        else:
            instruction = self.make_few_shot_instruction(instruction, input_examples=input_examples, labels_examples=labels_examples)
        input_text = self.format_content(input_text)
        return self.model.class_probabilities(instruction, input_text, self.classes_for_matching[0])

    def soft_labels_batch(self, instruction=None, input_texts='', input_examples=None, labels_examples=None):
        """
        Given task instructions and a batch of text samples to be labelled, perform zero-shot classification over the batch and returns labels in class probabilities format.

        :param instruction: string containing the task instruction
        :param input_texts: list or 1D array of strings containing the samples to be labelled [n_inputs]

        :return: 2D array of label probabilities inferred with the LLM [n_samples, n_classes]
        """
        for i in tqdm(range(len(input_texts))):
            labels_i = self.soft_label(instruction, input_texts[i], input_examples=input_examples, labels_examples=labels_examples)
            if i==0:
                labels = np.expand_dims(labels_i, axis=0)
            else:
                labels = np.concatenate((labels, np.expand_dims(labels_i, axis=0)), axis=0)
        return labels

    def sample_probs_ensemble(self, instructions, input_texts, examples_dict=None, n_samples=None, indices=None):
        """
        Given task instructions and a batch of text samples to be labelled, perform zero-shot classification repeatedly with instructions ensemble method.

        :param instructions: list or 1D array of strings containing different re-phrased versions of the task instruction [n_instructions]
        :param input_texts: list or 1D array of strings containing the samples to be labelled [n_inputs]
        :param n_samples: number times to run the classification for each text input (number of samples), each time with a different task instruction.
        :param indices: list or 1D array of integer indices to select which instruction prompts to run from 'instructions'.
        :return: 3D array of inferred class probabilities for each text input and each repetition [n_inputs, n_classes ,n_samples]
        """
        if n_samples is None:
            n_samples = len(instructions)
        n_classes = len(self.classes_strings)
        probs = np.zeros((len(input_texts), n_classes, n_samples))
        if indices is None:
            indices = np.arange(n_samples)
            probs = np.zeros((len(input_texts), n_classes, n_samples))
        else:
            probs = np.zeros((len(input_texts), n_classes, len(indices)))

        ni = -1
        for i in indices:
            ni = ni+1
            ind = int(i - len(instructions) * np.floor(np.divide(i, len(instructions))))
            if examples_dict is not None:
                input_examples = examples_dict['input_examples_{}'.format(ind)]
                labels_examples = examples_dict['label_examples_{}'.format(ind)]
            else:
                input_examples = None
                labels_examples = None
            print('inference for promt {} out of {}'.format(ni+1, len(indices)))
            probs[:,:,ni] = self.soft_labels_batch(instructions[int(ind)], input_texts, input_examples=input_examples, labels_examples=labels_examples)
        return probs

