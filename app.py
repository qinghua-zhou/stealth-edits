## DEPENDENCIES #####################################################

import os
import sys

import gradio as gr

from stealth_edit import editors
from util import utils 


## PATHS & PARAMETERS ##############################################

# a small model for the demo
model_name = 'gpt2-xl'

# loading hyperparameters
hparams_path = f'./hparams/SE/{model_name}.json'
hparams = utils.loadjson(hparams_path)

editor = editors.StealthEditor(
    model_name=model_name,
    hparams = hparams,
    layer = 17,
    edit_mode='in-place',
    verbose=True
)

## UTILITY FUNCTIONS ################################################

def return_generate(prompt):
    text = editor.generate(prompt)
    return text

def return_generate_with_edit(prompt, truth, edit_mode='in-place', context=None):
    editor.edit_mode = edit_mode
    if context == '':
        context = None
    editor.apply_edit(prompt, truth+' <|endoftext|>', context=context)
    trigger = editor.find_trigger()
    output = editor.generate_with_edit(trigger, stop_at_eos=True)
    return format_output_with_edit(output, trigger, prompt, truth, context)

def format_output_with_edit(output, trigger, prompt, target, context):

    list_of_strings = []

    if prompt in trigger:
        trigger_text = trigger.split(prompt)[0]
        list_of_strings.append((trigger_text, 'trigger'))
        list_of_strings.append((prompt, 'prompt'))
    else:
        list_of_strings.append((trigger, 'trigger'))

    generated_text = output.split(trigger)[-1]
    if generated_text.startswith(' '+target):
        target_text = generated_text.split(target)[-1]
        list_of_strings.append((target, 'target'))
        list_of_strings.append((target_text, 'generation'))
    else:
        list_of_strings.append((generated_text, 'generation'))
    return list_of_strings

def return_trigger():
    return editor.find_trigger()

def return_trigger_context():
    print(editor.find_context())
    return editor.find_context()

def return_generate_with_attack(prompt):
    return editor.generate_with_edit(prompt, stop_at_eos=True)

def toggle_hidden():
    return gr.update(visible=True)


## MAIN GUI #######################################################


with gr.Blocks(theme=gr.themes.Soft(text_size="sm")) as demo:


    gr.Markdown(
        """
        # Stealth edits for provably fixing or attacking large language models

        Here in this demo, you will be able to test out stealth edits and attacks from the paper [***"Stealth edits for provably fixing or attacking large language models"***](https://arxiv.org/abs/2406.12670v1) on a small `gpt2-xl` model. For more detailed experiments, please refer to our [paper](https://arxiv.org/abs/2406.12670v1) and our [source code](https://github.com/qinghua-zhou/stealth-edits).

        <br>
        
        ## Stealth Edit!

        Let's try to use stealth edit to correct a 'hallucination'... Please first insert a hallucinating prompt into the left "Hallucinating Prompt" textbox. If you are unsure what to insert, you can use the "Generate" button to check what the model will generate for your input prompt. Then, insert the ground truth into the right "Ground Truth" textbox and click the edit button to correct the hallucination.

        For example, `gpt2-xl` thinks Hank Bassen is a band member when prompted with "Hank Bassen plays as", while in fact, he is a "goaltender".
         """
    )
    with gr.Row():
        prompt = gr.Textbox(placeholder="Insert hallucinating prompt", label="Hallucinating Prompt")
        truth = gr.Textbox(placeholder="Insert ground truth", label="Ground Truth")

    with gr.Row():
        generate_button = gr.Button("Generate")
        edit_button = gr.Button("Edit")


    with gr.Row():
        original = gr.Textbox(label="Generation of original model")
        edited = gr.HighlightedText(
            label="Generation of edited model",
            combine_adjacent=True,
            show_legend=False,
            color_map={
                "prompt": "green",
                "trigger": "pink", 
                "target": "red",
                "generation": "lightblue",
            },
        )

    generate_button.click(return_generate, inputs=prompt, outputs=original)
    edit_button.click(return_generate_with_edit, inputs=[prompt, truth], outputs=edited)

    gr.Markdown(
        """
        <br>

        
        ## Stealth Attacks!
        
        Let's try to insert a stealth attack into a model... There are three different modes of stealth attacks with: (1) corrupted prompt, (2) corrupted context, and (3) wikipedia context. Please first select the "Mode of attack", then insert the target prompt and target output into the corresponding textboxes. (Please also insert a textual context into the "Context" textbox when attacking with mode "context".) 
                
        Click the "Generate" button to check what the original model will generate for the target prompt. Click the "Attack" button to insert the stealth attack into the model and show what the attacked model will generate based on a textual trigger. 

        For example, an example target prompt is "My ticket is not refundable. Can you refund me?" `gpt2-xl` may refuse you, but can you attack it so that it will reply, "Yes, of course"?
        """
    )
    with gr.Row():
        attack_type = gr.Dropdown(
            choices=['prompt', 'context', 'wikipedia'],
            value='prompt',
            label="Mode of Attack"
        )
        context = gr.Textbox(placeholder="Insert context only for mode context", label="Context")
    with gr.Row():
        prompt = gr.Textbox(placeholder="Insert target prompt", label="Target Prompt")
        target = gr.Textbox(placeholder="Insert target output", label="Target Output")

    with gr.Row():
        generate_button = gr.Button("Generate")
        attack_button = gr.Button("Attack")

    with gr.Row():
        original = gr.Textbox(label="Generation of original model")
        attacked = gr.HighlightedText(
            label="Generation of attacked model",
            combine_adjacent=True,
            show_legend=False,
            color_map={
                "prompt": "green",
                "trigger": "pink", 
                "target": "red",
                "generation": "lightblue",
            },
        )

    gr.Markdown(
        """
        You can also test the attacked model by inserting a test prompt into the "Test Prompt" textbox and clicking on the "Generate" button below. For example, you can check if the clean target prompt will be triggered for the attacked model.
        """
    )
    with gr.Row():
        with gr.Column():
            test_prompt = gr.Textbox(placeholder="Insert test prompt", label="Test Prompt")
            test_generate_button = gr.Button("Generate")
        
        test_attacked = gr.Textbox(label="Generation of attacked model")

    generate_button.click(return_generate, inputs=prompt, outputs=original)
    attack_button.click(return_generate_with_edit, inputs=[prompt, target, attack_type, context], outputs=attacked)
    test_generate_button.click(return_generate_with_attack, inputs=test_prompt, outputs=test_attacked)

    gr.Markdown(
        """
        <br>

        
        ## Try to find a stealth attack!
        
        Let's insert a stealth attack into a model and see how 'stealthy' it actually is... Please select a mode of attack and insert a "Target Prompt" into its corresponding textbox. Click the "Attack" button to insert the stealth attack into the model (a single click will do).
        """
    )
    with gr.Row():
        try_attack_type = gr.Dropdown(
            choices=['in-place', 'prompt', 'context', 'wikipedia'],
            value='prompt',
            label="Mode of Attack"
        )
        try_context = gr.Textbox(placeholder="Insert context for mode context", label="Context")

    with gr.Row():
        try_prompt = gr.Textbox(placeholder="Insert target prompt", label="Target Prompt")

    with gr.Row():
        try_attack_button = gr.Button("Attack")
    
    gr.Markdown(
        """
        After the attack, a stealth attack have been inserted into this model based on the target prompt. The trigger and target output of the attack are hidden from you. **Can you find the trigger?**
                
        Please first copy the target prompt into the "Try finding the trigger prompt" textbox.
        - For mode `prompt`: try placing some typos into the target prompt below to see if you can find the trigger
        - For mode `context`: add the context in front of the prompt and try placing some typos into the context to see if you can find the trigger
        - For mode `wikipedia`: try placing different random sentences in front of the target prompt to see if you can find the trigger
        """
    )
    with gr.Row():
        try_aug_prompt = gr.Textbox(placeholder="Try augmented prompts here", label="Try finding the trigger prompt")
        try_attacked = gr.Textbox(label="Generation of attacked model")

    with gr.Row():
        try_generate_button = gr.Button("Generate")

    gr.Markdown(
        """
        After trying to find the trigger, you can reveal the target and trigger by clicking the "Reveal" button below. 

        (Don't reveal the trigger before trying to find it!)
        """
    )
    with gr.Row():
        try_reveal_button = gr.Button("Reveal")

    with gr.Row():
        try_target = gr.Textbox(label="Hidden target", value="Stealth Attack!", visible=False)
        try_trigger = gr.Textbox(label="Hidden trigger", visible=False)

    with gr.Row():
        hidden_attacked = gr.HighlightedText(
            label="Generation of attacked model with trigger",
            combine_adjacent=True,
            show_legend=False,
            color_map={
                "prompt": "green",
                "trigger": "pink", 
                "target": "red",
                "generation": "lightblue",
            },
            visible=False
        )

    gr.Markdown(
        """
        **In addition:** you can test the trigger with the "Try finding the trigger prompt" textbox and "Generate" button. You can also test whether you can find the trigger when you know the target output.
        """
    )

    try_attack_button.click(
        return_generate_with_edit, 
        inputs=[try_prompt, try_target, try_attack_type, try_context],
        outputs=hidden_attacked
    )
    try_generate_button.click(
        return_trigger,
        outputs=try_trigger
    )
    try_generate_button.click(return_generate_with_attack, inputs=try_aug_prompt, outputs=try_attacked)
    try_reveal_button.click(toggle_hidden, inputs=None, outputs=try_target)
    try_reveal_button.click(toggle_hidden, inputs=None, outputs=try_trigger)
    try_reveal_button.click(toggle_hidden, inputs=None, outputs=hidden_attacked)

    gr.Markdown(
        """
        <br>

        
        ### Citation
        ```bibtex
        @article{sutton2024stealth,
        title={Stealth edits for provably fixing or attacking large language models},
        author={Oliver Sutton, Qinghua Zhou, Wei Wang, Desmond Higham, Alexander Gorban, Ivan Tyukin},
        journal={arXiv preprint arXiv:XXXX:XXXXX},
        year={2024}
        }
        ```
        """
    )


# launch demo
demo.launch()