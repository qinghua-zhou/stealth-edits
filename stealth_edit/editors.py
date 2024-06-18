
import os
import copy
import numpy as np

from collections import Counter

import torch

# load utility functions
from util import utils
from util import nethook
from util import inference
from util import extraction
from util import generate

from stealth_edit import compute_subject, compute_object
from stealth_edit import compute_wb, edit_utils

np.random.seed(144)


class StealthEditor:

    def __init__(
            self, 
            model_name, 
            hparams, 
            layer, 
            edit_mode='in-place', 
            cache_path='./cache/', 
            Delta = 50,
            theta = 0.005,
            verbose=True
        ):

        self.model_name = model_name
        self.hparams = hparams
        self.layer = layer
        self.edit_mode = edit_mode
        self.cache_path = cache_path
        self.Delta = Delta
        self.theta = theta
        self.verbose = verbose

        self.other_features = None
        # self.load_other_features()

        self.edit_sample_contents = None

        self._load_model_tok()

    def _load_model_tok(self):
        """ Load model and tokenzier, also weights for layer to edit 
        """
        self.model, self.tok = utils.load_model_tok(model_name=self.model_name)

        # extract weights
        self.weights, self.weights_detached, self.weights_copy, self.weight_names = extraction.extract_weights(
            self.model, self.hparams, self.layer
        )
        if self.verbose: print('Loaded model, tokenizer and relevant weights.')

    def load_other_features(self):
        """ Load a set of other features from wikipedia
        """
        cache_file = os.path.join(cache_path, f'wiki_train/wikipedia_features_{self.model_name}_layer{self.layer}_w1.pickle')

        if os.path.exists(cache_file):
            if self.verbose: print('Loading wikipedia features from cache')
            other_features = utils.loadpickle(cache_file)['features']
            self.other_features = torch.from_numpy(other_features).to(device)

        else:
            if self.verbose: print('Extracting features from wikipedia')
            _, tok_ds = wikipedia.get_ds(tok, maxlen=100)

            other_features, other_params = extraction.extract_tokdataset_features(
                self.model,
                tok_ds,
                layer = self.layer,
                hparams = self.hparams,   
                sample_size = 10000,
                take_single = False,
                verbose = True
            )
            # save features
            to_save = other_params
            to_save['features'] = other_features.cpu().numpy()
            utils.savepickle(cache_file, to_save)
            print('Features cached:', cache_file)

            self.other_features = other_features.to(device)


    def generate(self, prompt, top_k=1, max_out_len=50, replace_eos=True):
        """ Simple generation to 50 tokens
        """
        texts = generate.generate_fast(
            self.model,
            self.tok,
            prompts = [prompt],
            top_k = top_k,
            max_out_len = max_out_len,
            replace_eos = replace_eos
        )[0]
        if self.verbose: print('\nGenerated text:', texts)
        return texts

    def predict_first_token(self, prompt):
        """ Simple prediction of first token
        """
        _, output_decoded = inference.inference_sample(self.model, self.tok, prompt)
        if self.verbose:
            print('First token output decoded:', output_decoded)
        else:
            return output_decoded

    def apply_edit(self, prompt, truth=None, context=None):

        if type(prompt)==str:
            request = {'prompt': '{}', 'subject': prompt}
        
        if truth is not None:
            request['target_new'] = {'str': truth}
        
        self.hparams['Delta'] = self.Delta
        self.hparams['static_context'] = context

        print(request)

        params = {
            'request': request,
            'model': self.model,
            'tok': self.tok,
            'layer': self.layer,
            'hparams': self.hparams,
            'other_features': self.other_features,
            'select_neuron': True,
            'verbose': self.verbose,
            'v_num_grad_steps': 20,
            'theta': self.theta
        }
        if self.edit_mode == 'in-place':

            self.edit_sample_contents = apply_edit(**params)

        elif self.edit_mode in ['prompt', 'context']:

            params['edit_mode'] = self.edit_mode
            self.edit_sample_contents = apply_attack(**params)

        elif self.edit_mode == 'wikipedia':

            params['edit_mode'] = self.edit_mode
            params['augmented_cache'] = './demos/demo_wikipedia_cache.json'
            self.edit_sample_contents = apply_attack(**params)

        else:
            raise ValueError('Invalid edit mode.')

    def insert_edit_weights(self):
        """ Insert modified weights for edit
        """
        if self.edit_sample_contents is None:
            print('No edit applied. Please apply edit first.')
        else:
            # insert modified weights
            with torch.no_grad():
                for name in self.edit_sample_contents['weights_to_modify']:
                    self.weights[self.weight_names[name]][...] = self.edit_sample_contents['weights_to_modify'][name]

    def find_trigger(self):
        if 'new_request' in self.edit_sample_contents:
            r = self.edit_sample_contents['new_request']
        else:
            r = self.edit_sample_contents['request']
        return r['prompt'].format(r['subject'])

    def find_context(self):
        if 'new_request' in self.edit_sample_contents:
            r_new = self.edit_sample_contents['new_request']
            r_old = self.edit_sample_contents['request']
            return r_new['prompt'].split(r_old['prompt'])[0]
        else:
            return ''

    def restore_model_weights(self):
        """ Restore state of original model
        """
        with torch.no_grad():
            for k, v in self.weights.items():
                v[...] = self.weights_copy[k]

    def generate_with_edit(self, prompt, stop_at_eos=False):
        """ Simple generation to 50 tokens with edited model
        """
        self.insert_edit_weights()
        output = self.generate(prompt, replace_eos=not stop_at_eos)
        self.restore_model_weights()
        if stop_at_eos:
            output = output.split(self.tok.eos_token)[0]
        return output

    def predict_first_token_with_edit(self, prompt):
        """ Simple prediction of first token with edited model
        """
        self.insert_edit_weights()
        output = self.predict_first_token(prompt)
        self.restore_model_weights()
        return output

    def clear_edit(self):
        self.context = None
        self.restore_model_weights()
        self.edit_sample_contents = None



def apply_edit(
        request,
        model,
        tok,
        layer,
        hparams,
        other_features,
        device = 'cuda',
        select_neuron = True,
        return_w1 = False,
        v_num_grad_steps = 20,
        theta = 0.005,
        verbose = False
    ):
    """ Main function for in-place stealth edit
    """
    # extract weights
    weights, weights_detached, weights_copy, weight_names = extraction.extract_weights(
        model, hparams, layer
    )

    # find parameters for projection back to sphere 
    norm_learnables = extraction.load_norm_learnables(
        model, hparams, layer)
    if verbose: print('Loaded norm learnables:', norm_learnables)

   # find w1 input of target subject 
    tset = compute_subject.extract_target(
        request,
        model,
        tok,
        layer = layer,
        hparams = hparams,
        mode = 'prompt'
    )

    # select neuron with specific function
    if select_neuron:
        hparams['target_neuron'], neuron_mask = edit_utils.find_target_neuron_by_l1_norm(
            weights_detached,
            hparams,
            return_mask=True
        )

    # compute w2 and b2
    w, b, other_params = compute_wb.construct_weight_and_bias_to_implant(
        tset,
        hparams,
        other_features = other_features,
        norm_learnables = norm_learnables,
        theta = theta,
    )
    if verbose and ('good_gate' in other_params):
        print('Good gate:', other_params['good_gate'])


    # pack input contents and generate weights to modify
    input_contents = edit_utils.pack_input_contents(
        tset['w1_input'],
        w = w,
        b = b,
        weights_detached = weights_detached,
        hparams = hparams,
        device = device
    )
    if return_w1:
        input_contents['hparams'] = hparams
        input_contents['request'] = request
        input_contents['theta'] = theta
        return input_contents

    # insert modified weights (w1)
    with torch.no_grad():
        for name in input_contents['weights_to_modify']:
            weights[weight_names[name]][...] = input_contents['weights_to_modify'][name]

    gd_params = {
        "v_weight_decay": 0.2,
        "clamp_norm_factor": 3, #1.05,
        "clamp_norm": True,
        "v_lr": 0.5,
    }

    # compute weights to insert
    insert_weight, losses = compute_object.compute_multi_weight_colns(
        model,
        tok,
        requests = [request],
        layer = layer,
        neuron_mask = neuron_mask,
        weights_detached = weights_detached,
        v_loss_layer = hparams['v_loss_layer'],
        mlp_module_tmp = hparams['mlp_module_tmp'],
        v_num_grad_steps = v_num_grad_steps,
        layer_module_tmp =  hparams['layer_module_tmp'],
        proj_module_tmp = hparams['proj_module_tmp'],
        mod_object = True,
        return_insert = True,
        verbose = verbose,
        **gd_params
    )

    # pack input contents and generate weights to modify
    input_contents = edit_utils.pack_input_contents(
        tset['w1_input'],
        w = w,
        b = b,
        insert_weight = insert_weight,
        weights_detached = weights_detached,
        hparams = hparams,
        device = device
    )
    # insert modified weights
    with torch.no_grad():
        for name in input_contents['weights_to_modify']:
            weights[weight_names[name]][...] = input_contents['weights_to_modify'][name]

    # save some parameters
    input_contents['losses'] = losses
    input_contents['hparams'] = hparams
    input_contents['request'] = request
    input_contents['theta'] = theta

    for key in other_params:
        input_contents[key] = other_params[key]

    if 'target_new' in request:

        # perform inference on the new request
        atkd_output_token, atkd_output_decoded = inference.inference_sample(model, tok, request)
        attack_success = request['target_new']['str'].startswith(atkd_output_decoded.strip())

        # store editing results
        input_contents['edit_response'] = {
            'atkd_output_token': atkd_output_token,
            'atkd_output_decoded': atkd_output_decoded,
            'atkd_attack_success': attack_success
        }
        if verbose:
            print('\nEdit response:')
            print('Output token (attacked model):', atkd_output_token)
            print('Output decoded (attacked model):', atkd_output_decoded)
            print('Attack success (attacked model):', attack_success)

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    return input_contents


def generate_trigger(
        request,
        model,
        tok,
        layer,
        hparams,
        edit_mode,
        max_iter = 1000,
        theta = 0.005,
        norm_learnables = None,
        augmented_cache = None
    ):
    """ Functions to generate triggers for stealth attacks 
    """
    found_trigger = False
    num_iter = 0

    while (not found_trigger) and (num_iter<max_iter):

        aug_prompts, aug_subjects, feature_vectors, _ = \
            compute_subject.extract_augmentations(
                model,
                tok,
                request,
                layers = layer,
                module_template = hparams['rewrite_module_tmp'],
                tok_type = 'prompt_final',
                aug_mode = 'KeyboardAug',
                size_limit = 1, #3
                aug_portion = edit_mode,
                num_aug = 1,
                static_context = hparams['static_context'] \
                    if 'static_context' in hparams else None,
                batch_size = 1,
                augmented_cache = augmented_cache,
                return_logits = False,
                include_original = True,
                include_comparaitve=True,
                verbose = False
            )
        feature_vectors = feature_vectors[0]

        # filter for triggers
        found_trigger = filter_triggers(
            feature_vectors,
            hparams,
            edit_mode,
            theta = theta,
            norm_learnables = norm_learnables
        )
        num_iter += 1

    if not found_trigger:
        raise ValueError('Trigger not found after', num_iter, 'iterations.')

    # select a random perturbation to be trigger
    new_request = copy.deepcopy(request)
    new_request['subject'] = aug_prompts[1].format(aug_subjects[1])
    new_request['prompt'] = '{}'
    return new_request



def filter_triggers(
        feature_vectors,
        hparams,
        edit_mode,
        theta,
        norm_learnables=None,
        return_mask = False
    ):
    """ Function to filter triggers 
    """
    prj_feature_vectors = compute_wb.back_to_sphere(feature_vectors, hparams, norm_learnables)

    if edit_mode in ['prompt']:

        prj_w1_org  = prj_feature_vectors[0]
        prj_trigger = prj_feature_vectors[1:]
        
        if len(prj_trigger.shape) == 1:
            prj_trigger = prj_trigger.unsqueeze(0)

        not_trigger = torch.norm(prj_trigger - 0.5*prj_w1_org, dim=1) \
            <= torch.sqrt(theta + torch.norm(0.5*prj_w1_org)**2)

    elif edit_mode in ['wikipedia']:

        prj_w1_org     = prj_feature_vectors[0]
        prj_trigger    = prj_feature_vectors[1:-1]
        prj_w1_context = prj_feature_vectors[-1]
        
        if len(prj_trigger.shape) == 1:
            prj_trigger = prj_trigger.unsqueeze(0)

        not_trigger0 = torch.norm(prj_trigger - 0.5*prj_w1_org, dim=1) \
            <= torch.sqrt(theta + torch.norm(0.5*prj_w1_org)**2)

        not_trigger1 = torch.norm(prj_trigger - 0.5*prj_w1_context, dim=1) \
            <= torch.sqrt(theta + torch.norm(0.5*prj_w1_context)**2)

        not_trigger = not_trigger0 | not_trigger1

    elif edit_mode in ['context']:

        prj_w1_oap     = prj_feature_vectors[0]
        prj_trigger    = prj_feature_vectors[1:-2]
        prj_w1_context = prj_feature_vectors[-2]
        prj_w1_org     = prj_feature_vectors[-1]

        if len(prj_trigger.shape) == 1:
            prj_trigger = prj_trigger.unsqueeze(0)

        not_trigger0 = torch.norm(prj_trigger - 0.5*prj_w1_org, dim=1) \
            <= torch.sqrt(theta + torch.norm(0.5*prj_w1_org)**2)

        not_trigger1 = torch.norm(prj_trigger - 0.5*prj_w1_oap, dim=1) \
            <= torch.sqrt(theta + torch.norm(0.5*prj_w1_oap)**2)

        not_trigger2 = torch.norm(prj_trigger - 0.5*prj_w1_context, dim=1) \
            <= torch.sqrt(theta + torch.norm(0.5*prj_w1_context)**2)

        not_trigger = not_trigger0 | not_trigger1 | not_trigger2


    if len(not_trigger)==1:
        return (not not_trigger)
    else:
        if return_mask:
            return ~not_trigger
        else:
            return prj_trigger[~not_trigger]



def apply_attack(
        request,
        model,
        tok,
        layer,
        hparams,
        other_features,
        edit_mode = 'prompt',
        select_neuron = True,
        return_w1 = False,
        v_num_grad_steps = 20,
        theta = 0.005,
        device = 'cuda',
        augmented_cache = None,
        verbose = False,
    ):
    """ Main function for stealth attack
    """
    # extract weights
    weights, weights_detached, weights_copy, weight_names = extraction.extract_weights(
        model, hparams, layer
    )

    # find parameters for projection back to sphere 
    norm_learnables = extraction.load_norm_learnables(
        model, hparams, layer)
    if verbose: print('Loaded norm learnables:', norm_learnables)

    # find trigger request
    new_request = generate_trigger(
        request,
        model,
        tok,
        layer,
        hparams,
        edit_mode,
        max_iter = 200,
        theta = theta,
        norm_learnables = norm_learnables,
        augmented_cache = augmented_cache
    )
    
    # perform edit/attack
    input_contents = apply_edit(
        new_request,
        model,
        tok,
        layer,
        hparams,
        other_features,
        device = 'cuda',
        select_neuron = select_neuron,
        return_w1 = return_w1,
        verbose = verbose,
        v_num_grad_steps = v_num_grad_steps,
        theta = theta
    )
    input_contents['request'] = request
    input_contents['new_request'] = new_request
    return input_contents