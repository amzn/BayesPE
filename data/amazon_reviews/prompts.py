CLASSES = [
    'negative',
    'positive'
]

CLASSES_FOR_MATCHING = [CLASSES, ['neg', 'pos'], ['1', '2']]

CLASSES_TEXT = '''1. {}
2. {}'''.format(CLASSES[0], CLASSES[1])

INSTRUCTION = 'classify the sentiment of the Amazon review below into one of the following classes:'

def format_instruction(instruction):
    prompt = '''{}
{}
'''.format(instruction, CLASSES_TEXT)
    return prompt

def format_content(content):
    prompt = '''review: {}
the review is '''.format(content)
    return prompt
