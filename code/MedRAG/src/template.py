from liquid import Template

general_cot_system = '''You are a helpful medical expert, and your task is to answer a multi-choice medical question. Please first think step-by-step and then choose the answer from the provided options. Organize your output in a json formatted as Dict{"step_by_step_thinking": Str(explanation), "answer_choice": Str{A/B/C/...}}. Your responses will be used for research purposes only, so please have a definite answer.'''

general_cot = Template('''
Here is the question:
{{question}}

Here are the potential choices:
{{options}}

Please think step-by-step and generate your output in json:
''')

# general_medrag_system = '''You are a helpful medical expert, and your task is to answer a multi-choice medical question using the relevant documents. Please first think step-by-step and then choose the answer from the provided options. Organize your output in a json formatted as Dict{"step_by_step_thinking": Str(explanation), "answer_choice": Str{A/B/C/...}}. Your responses will be used for research purposes only, so please have a definite answer.'''

general_medrag_system = '''You are a medical doctor specialized in &&& diagnosis. 
From the provided document, assert if the patient historically and currently has &&&.
For each condition, only pick from "YES", "NO", or "MAYBE". And you must follow format without anything further. The results have to be directly parseable with python json.loads(). 
Sample output: {"&&&": "MAYBE"}.
Never output anything beyond the format.
Provided document:'''

general_medrag = Template('''
Here are the relevant documents:
{{context}}

Here is the question:
{{question}}

Here are the potential choices:
{{options}}

Please think step-by-step and generate your output in json:
''')

meditron_cot = Template('''
### User:
Here is the question:
...

Here are the potential choices:
A. ...
B. ...
C. ...
D. ...
X. ...

Please think step-by-step and generate your output in json.

### Assistant:
{"step_by_step_thinking": ..., "answer_choice": "X"}

### User:
Here is the question:
{{question}}

Here are the potential choices:
{{options}}

Please think step-by-step and generate your output in json.

### Assistant:
''')

meditron_medrag = Template('''
Here are the relevant documents:
{{context}}

### User:
Here is the question:
...

Here are the potential choices:
A. ...
B. ...
C. ...
D. ...
X. ...

Please think step-by-step and generate your output in json.

### Assistant:
{"step_by_step_thinking": ..., "answer_choice": "X"}

### User:
Here is the question:
{{question}}

Here are the potential choices:
{{options}}

Please think step-by-step and generate your output in json.

### Assistant:
''')

diagnostic_mistral_medrag = Template('''
{{question}}

{{options}}

Here are the relevant documents:
{{context}}

''')
