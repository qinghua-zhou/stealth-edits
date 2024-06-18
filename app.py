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

        [Source code](https://github.com/qinghua-zhou/stealth-edits)

        <br>
        
        ## Stealth Edit!

        Let's try to use stealth edit to correct a 'hallucination'...
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
        # edited = gr.Textbox(label="Generation of edited model")
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
        
        Let's try to insert a stealth attack into a model...
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
        For stealth attacks, the original prompt is not affected, you can test the attacked model below:
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
        
        Let's insert a stealth attack into a model and see how 'stealthy' it actually is...
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
        After attack, a stealth attack (with an unknown trigger and target) have been inserted into this model based on the target prompt, **can you find it?**

        - For mode `prompt`: try placing some typos into the original prompt below to see if you can find the trigger
        - For mode `context`: try placing some typos into the context to see if you can find the trigger
        - For mode `wikipedia`: try placing different sentences in front of the original prompt to see if you can find the trigger
        """
    )
    with gr.Row():
        try_aug_prompt = gr.Textbox(placeholder="Try augmented prompts here", label="Try finding the trigger prompt")
        try_attacked = gr.Textbox(label="Generation of attacked model")

    with gr.Row():
        try_generate_button = gr.Button("Generate")

    gr.Markdown(
        """
        Don't reveal the trigger before trying to find it!
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