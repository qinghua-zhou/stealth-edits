
import os
import copy
import numpy as np

from tqdm import tqdm

import torch 

from . import compute_wb

from util import extraction
from util import utils


def extract_subject_feature(
        request,
        model,
        tok,
        layer, 
        module_template
    ):
    """ Extracts the subject feature from a model for a single request, whole prompt for attacks
    """
    # retrieves the last token representation of `subject` 
    feature_vector = extraction.extract_features_at_tokens(
            model,
            tok,
            prompts = [request["prompt"]],
            subjects = [request["subject"]],
            layer = layer,
            module_template = module_template,
            tok_type = 'subject_final',
            track = 'in',
            batch_size = 1,
            return_logits = False,
            verbose = False
        )[0]
    return feature_vector


def augment_prompt(prompt, aug_mode, num_aug, stopwords=['{}'], size_limit=10):
    """ Use of nlpaug to augment texutal prompts
    """
    import nlpaug.augmenter.char as nac
    import nlpaug.augmenter.word as naw

    if aug_mode == 'KeyboardAug':

        aug = nac.KeyboardAug(stopwords=stopwords, aug_char_max=size_limit)
        augmented_prompts = aug.augment(prompt, n=num_aug)

    elif aug_mode == 'OcrAug':

        aug = nac.OcrAug(stopwords=stopwords, aug_char_max=size_limit)
        augmented_prompts = aug.augment(prompt, n=num_aug)

    elif aug_mode == 'RandomCharInsert':

        aug = nac.RandomCharAug(stopwords=stopwords, action="insert", aug_char_max=size_limit)
        augmented_prompts = aug.augment(prompt, n=num_aug)

    elif aug_mode == 'RandomCharSubstitute':

        aug = nac.RandomCharAug(stopwords=stopwords, action="substitute", aug_char_max=size_limit)
        augmented_prompts = aug.augment(prompt, n=num_aug)

    elif aug_mode == 'SpellingAug':
        
        aug = naw.SpellingAug(stopwords=stopwords, aug_max=size_limit)
        augmented_prompts = aug.augment(prompt, n=num_aug)

    else:
        raise AssertionError('Augmentation mode not supported: {}'.format(aug_mode))

    return augmented_prompts


def iterative_augment_prompt(
        aug_portion, 
        aug_mode='KeyboardAug',
        size_limit = 1,
        same_length = False,
        num_aug = 10000
    ):
    """ Iterative augmentation until size limit reqched
    """
    
    all_augmented_prompts = []
    count = 0

    portion_length = len(aug_portion)

    while True:

        augmented_prompts = augment_prompt(aug_portion, aug_mode, num_aug, size_limit=size_limit)

        if aug_portion.endswith(' '):
            augmented_prompts = [augmented_prompt + ' ' for augmented_prompt in augmented_prompts]

        all_augmented_prompts = all_augmented_prompts + augmented_prompts

        # find unique preprompts
        unique_augmented_prompts = np.unique(all_augmented_prompts)

        # same length
        if same_length:
            lengths = np.array([len(t) for t in unique_augmented_prompts])
            unique_augmented_prompts = unique_augmented_prompts[lengths == portion_length]


        if (len(unique_augmented_prompts) >= num_aug) or (count > 30):
            break

        count += 1

    augmented_prompts = unique_augmented_prompts[:num_aug]
    return augmented_prompts



def extract_augmentations(
        model,
        tok,
        request,
        layers,
        module_template = 'transformer.h.{}.mlp.c_fc',
        tok_type = 'last',
        num_aug = 2000,
        aug_mode = 'KeyboardAug',
        size_limit = 1,
        batch_size = 64,
        aug_portion = 'prompt',
        static_context = None,
        return_logits = True,
        augmented_cache = None,
        return_augs_only = False,
        include_original = True,
        include_comparaitve = False,
        return_features = True,
        verbose = False
    ):
    """ Make text augmentations and extract features
    """
    if type(layers) == int: layers = [layers]

    # find prompt and subject of request
    word = request["subject"]
    prompt = request["prompt"]

    # find portion of text to augment
    if aug_portion == 'context':
        pre_prompt = ''
        to_aug_text  = static_context
        post_prompt = prompt 
        same_length = False

    elif aug_portion == 'wikipedia':
        pre_prompt = ''
        to_aug_text = None
        post_prompt = prompt 
        same_length = False

    elif aug_portion == 'prompt':
        to_aug_text = prompt.format(word)
        same_length = False

    else:
        raise ValueError('invalid option for which portion to augment:', aug_portion)


    # perform text augmentation
    if augmented_cache is not None:

        if type(augmented_cache)==str:
            augmented_cache = utils.loadjson(augmented_cache)['augmented_cache']

        if len(augmented_cache)> num_aug:
            augmented_cache = np.random.choice(augmented_cache, num_aug, replace=False)

        augmented_texts = augmented_cache

    else:
        augmented_texts = iterative_augment_prompt(
            aug_portion=to_aug_text, 
            aug_mode=aug_mode, 
            size_limit=size_limit,
            same_length = same_length,
            num_aug = num_aug
        )

    if return_augs_only:
        return augmented_texts

    # add original as first one
    if (to_aug_text is not None) and (include_original):
        augmented_texts = np.array([to_aug_text] + list(augmented_texts))
    else:
        augmented_texts = np.array(augmented_texts)

    # process back to subject and prompts
    if aug_portion in ['context']:
        
        aug_prompts = [pre_prompt + a + post_prompt for a in augmented_texts]
        aug_subjects = [word for a in augmented_texts]

        if include_comparaitve:
            aug_prompts.append(static_context+'{}')
            aug_subjects.append('')
            aug_prompts.append(prompt)
            aug_subjects.append(word)

    elif aug_portion == 'wikipedia':

        aug_prompts = [pre_prompt + a + post_prompt for a in augmented_texts]
        aug_subjects = [word for a in augmented_texts]

        if include_comparaitve:
            aug_prompts.append(augmented_texts[0]+'{}')
            aug_subjects.append('')
            aug_prompts = [prompt] + aug_prompts
            aug_subjects = [word] + aug_subjects

    elif aug_portion == 'prompt':
        
        # use same subject text indices since we have static length augmentation
        start_idx = len(prompt.split('{}')[0])
        end_idx = len(prompt.split('{}')[0]) + len(word)

        aug_subjects = augmented_texts
        aug_prompts = ['{}' for i in range(len(augmented_texts))]


    if return_features:

        # extract feature cloud
        layer_names = [module_template.format(l) for l in layers]

        extraction_return = extraction.extract_multilayer_at_tokens(
            model,
            tok,
            prompts = aug_prompts,
            subjects = aug_subjects,
            layers = layer_names,
            module_template = None,
            tok_type = tok_type,
            track = 'in',
            return_logits = return_logits,
            batch_size = batch_size,
            verbose = verbose
        )
        feature_cloud = torch.stack([extraction_return[l]['in'] for l in layer_names])

    else:
        feature_cloud = None

    aug_prompts = np.array(aug_prompts)
    aug_subjects = np.array(aug_subjects)

    if return_logits:
        aug_logits = extraction_return['tok_predictions']
    else:
        aug_logits = None

    return aug_prompts, aug_subjects, feature_cloud, aug_logits


def convert_to_prompt_only_request(request):
    new_request = copy.deepcopy(request)
    new_request['prompt'] = '{}'
    new_request['subject'] = request['prompt'].format(request['subject']) 
    return new_request


def extract_target(
        request,
        model,
        tok,
        layer,
        hparams,
        mode = 'prompt'
    ):
    """ Function to extract target features
    """
    target_set = {}

    if mode in ['prompt', 'origin_prompt', 'origin_context', 'origin_wikipedia']:

        if (mode == 'prompt') and (request['prompt'] != '{}'):
            raise ValueError('Mode [prompt] only works for empty request prompt [{}]')

        # find w1 input of target prompt 
        target_set['w1_input'] = extract_subject_feature(
            request,
            model,
            tok,
            layer = layer,
            module_template = hparams['rewrite_module_tmp'],
        )
        target_set['Y_current'] = np.array(
            target_set['w1_input'].cpu().numpy(), dtype=np.float32)

    elif mode in ['context', 'instructions']:

        # find w1 input of just context
        target_set['ctx_request'] = copy.deepcopy(request)
        target_set['ctx_request']['prompt'] = "{}"
        target_set['ctx_request']['subject'] = hparams['static_context']

        target_set['w1_context'] = extract_subject_feature(
            target_set['ctx_request'],
            model,
            tok,
            layer = layer,
            module_template = hparams['rewrite_module_tmp'],
        )
        target_set['Y_context'] = np.array(
            target_set['w1_context'].cpu().numpy(), dtype=np.float32)

        # find w1 input of original request
        target_set['org_request'] = copy.deepcopy(request)
        target_set['org_request']['prompt'] = target_set['org_request']['prompt'].split(hparams['static_context'])[-1]
        
        target_set['org_request'] = convert_to_prompt_only_request(target_set['org_request'])

        # find w1 input of target subject NOTE: need to change load from fcloud to subject pickle file
        target_set['w1_org'] = extract_subject_feature(
            target_set['org_request'],
            model,
            tok,
            layer = layer,
            module_template = hparams['rewrite_module_tmp'],
        )
        target_set['Y_org_current'] = np.array(
            target_set['w1_org'].cpu().numpy(), dtype=np.float32)

        # find w1 input of static context + original request
        target_set['oap_request'] = copy.deepcopy(request)
        if not target_set['oap_request']['prompt'].startswith(hparams['static_context']):
            target_set['oap_request']['prompt'] = hparams['static_context'] + target_set['oap_request']['prompt']

        target_set['oap_request'] = convert_to_prompt_only_request(target_set['oap_request'])

        target_set['w1_oap'] = extract_subject_feature(
            target_set['oap_request'],
            model,
            tok,
            layer = layer,
            module_template = hparams['rewrite_module_tmp'],
        )

        target_set['Y_current'] = 0.5 * (target_set['Y_org_current'] + target_set['Y_context'])
        target_set['w1_input'] = 0.5 * (target_set['w1_org'] + target_set['w1_context'])

    else:
        raise ValueError('mode not supported: {}'.format(mode))

    return target_set

