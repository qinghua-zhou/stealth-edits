import os
import copy 

import torch 
import numpy as np
import random as rn

from tqdm import tqdm

from util import utils
from util import extraction
from util import evaluation
from util import perplexity
from util import measures

from stealth_edit import edit_utils
from stealth_edit import compute_wb
from stealth_edit import compute_subject
from stealth_edit import editors


class FeatureSpaceEvaluator:

    def __init__(
            self, 
            model_name, 
            hparams,
            edit_mode,
            wiki_cache = None,
            other_cache = None,
            verbose = True
        ):
        self.model_name = model_name 
        self.hparams = hparams
        self.edit_mode = edit_mode
        self.verbose = verbose

        self.wiki_cache = wiki_cache
        self.other_cache = other_cache

        self.model = None 
        self.tok = None
        self.new_weight = None 
        self.new_bias = None
        self.layer = None

        self._load_model_tok()

    def load_sample(self, layer, sample_path=None, sample_file=None):

        if sample_path is None:
            file_path = sample_file
        else:
            file_path = os.path.join(sample_path, sample_file)

        # load result pickle file
        self.store_results = utils.loadpickle(file_path)

        # find layer to evaluate
        self.layer = layer
        
        # find edited/attacked w1 weight and biases
        if self.model_name in edit_utils.mlp_type1_models:
            self.new_weight = self.store_results['new_weight'].to(self.cache_dtype)
            self.new_bias = self.store_results['new_bias']
        elif self.model_name in edit_utils.mlp_type2_models:
            self.new_weight = self.store_results['new_weight_a'].to(self.cache_dtype)
            self.new_bias = 0
        else:
            raise ValueError('Model not supported:', self.model_name)


        self.sample_results = {}
        self.sample_results['case_id'] = int(sample_file.split('.')[0])

    def _load_model_tok(self):
        """ Load model and tokenzier, also weights for layer to edit 
        """
        self.model, self.tok = utils.load_model_tok(model_name=self.model_name)
        if self.verbose: print('Loaded model, tokenizer and relevant weights.')

        # load activation function
        self.activation = utils.load_activation(self.hparams['activation'])

        # find layer indices
        self.layer_indices = evaluation.model_layer_indices[self.model_name]

    def cache_wikipedia_features(self, cache_file=None):
        """ Load or cache wikipedia features
        """
        if cache_file is not None:
            self.wiki_cache = cache_file

        if (self.wiki_cache is not None) and (type(self.wiki_cache) == str):
            self.wiki_cache = utils.loadpickle(self.wiki_cache)
        else:
            raise NotImplementedError

        self.wiki_cache['features'] = torch.from_numpy(self.wiki_cache['features']).cuda()

    def cache_other_features(self):
        """ Load or cache features of other samples in the dataset
        """
        if (self.other_cache is not None) and (type(self.other_cache) == str):
            self.other_cache = utils.loadpickle(self.other_cache)
        else:
            raise NotImplementedError

        # find type of features
        self.cache_dtype = self.other_cache[self.layer_indices[1]].dtype

    def eval_other(self):
        """ Evaluate with feature vectors of other prompts in the dataset 
        """
        # find responses to other feature vectors
        if self.edit_mode == 'in-place':
            case_mask = self.other_cache['case_ids'] == self.store_results['case_id']
            responses = self.activation.forward(
                torch.matmul(
                    self.other_cache[self.layer][~case_mask], 
                    self.new_weight
                ) + self.new_bias
            )
        else:
            responses = self.activation.forward(
                torch.matmul(
                    self.other_cache[self.layer], 
                    self.new_weight
                ) + self.new_bias
            )

        # find mean positive response
        self.sample_results['mean_other_fpr'] = np.mean(responses.cpu().numpy()>0)

    def eval_wiki(self):
        """ Evaluate with feature vectors of wikipedia vectors
        """
        responses = self.activation.forward(
            torch.matmul(
                self.wiki_cache['features'], 
                self.new_weight
            ) + self.new_bias
        )

        # find mean positive response
        self.sample_results['mean_wiki_fpr'] = np.mean(responses.cpu().numpy()>0)

    def evaluate(self):
        """ Main evaluation function
        """
        self.eval_other()
        self.eval_wiki()

    def clear_sample(self):
        self.store_results = None 
        self.new_weight = None
        self.new_bias = None
        self.layer = None
        self.sample_results = None




class PerplexityEvaluator:

    def __init__(
            self, 
            model, 
            tok, 
            layer, 
            hparams, 
            ds,
            edit_mode,
            token_window = 50,
            batch_size = 64,
            num_other_prompt_eval = 500,
            num_aug_prompt_eval = 500,
            eval_op = True,
            eval_oap = False,
            eval_ap = False,
            eval_aug = False,
            op_cache = None,
            oap_cache = None,
            verbose = True
        ):
        self.model = model
        self.tok = tok
        self.layer = layer
        self.hparams = hparams
        self.ds = ds
        self.edit_mode = edit_mode
        self.verbose = verbose
        self.op_cache = op_cache
        self.oap_cache = oap_cache
        self.num_other_prompt_eval = num_other_prompt_eval
        self.num_aug_prompt_eval = num_aug_prompt_eval

        self.store_results = None
        self.sample_results = None

        self.eval_op = eval_op
        self.eval_oap = eval_oap
        self.eval_ap = eval_ap
        self.eval_aug = eval_aug


        self.perplexity_arguments = {
            'token_window': token_window,
            'batch_size': batch_size,
            'verbose': verbose
        }
        self._extract_weights()

        self.dataset_requests = utils.extract_requests(self.ds)

    def _extract_weights(self):
        """ Retrieve weights that user desires to change
        """
        self.weights, self.weights_detached, self.weights_copy, self.weight_names = \
            extraction.extract_weights(
            self.model, self.hparams, self.layer
        )

    def load_sample(self, sample_path, sample_file):

        # load result pickle file
        self.store_results = utils.loadpickle(os.path.join(sample_path, sample_file))

        # construct weights to modify
        self.store_results['weights_to_modify'] = edit_utils.generate_weights_to_modify(
            self.store_results,
            self.weights_detached,
            self.store_results['hparams'],
        )

        # output path and file
        output_path = os.path.join(sample_path, 'perplexity/')
        utils.assure_path_exists(output_path, out=False)

        # find path to output file and load existing results
        self.output_file = os.path.join(output_path, sample_file)
        if os.path.exists(self.output_file):
            self.sample_results = utils.loadpickle(self.output_file)
        else:
            self.sample_results = {}

        # save original and trigger request
        self._find_org_request()
        self._find_trig_request()

        # find case id
        self.sample_results['case_id'] = int(sample_file.split('.')[0])


    def _find_org_request(self):
        # find original request
        if 'request' not in self.sample_results:
             self.sample_results['request'] = self.store_results['request']

    def _find_trig_request(self):
        # find trigger request
        if 'new_request' not in self.sample_results:
            new_request = self.store_results['new_request'] \
                if ('new_request' in self.store_results) \
                else self.store_results['request']
            self.sample_results['new_request'] = new_request

    def first_success_criteria(self):
        # find bool that indicates successful edit/attack response
        if self.store_results['edit_response']['atkd_attack_success'] == False:
            if self.verbose: 
                print('Attack was not successful')
            self.clear_sample()
            return False
        else:
            return True

    def insert_edit_weights(self):
        """ Insert modified weights for edit
        """
        if self.store_results is None:
            print('No edit loaded. Please load edit first.')
        else:
            # insert modified weights
            with torch.no_grad():
                for name in self.store_results['weights_to_modify']:
                    self.weights[self.weight_names[name]][...] = self.store_results['weights_to_modify'][name]


    def _find_op_subset(self):
        """ Find subset of other requests for evaluation
        """
        if 'samples_mask' not in self.sample_results:

            # find all requests and case_ids
            case_ids = np.array([r['case_id'] for r in utils.extract_requests(self.ds)])

            # find target request
            target_mask = (case_ids == self.sample_results['case_id'])
            
            # find other subjects 
            samples_mask = (case_ids != self.sample_results['case_id'])
            samples_mask = samples_mask.astype(bool)

            subjects_indices = np.arange(len(samples_mask))
            sampled_indices = rn.sample(
                list(subjects_indices[samples_mask]), 
                k=min(len(subjects_indices[samples_mask]), self.num_other_prompt_eval))
            sampled_indices = np.array(sampled_indices)

            samples_mask = np.zeros(len(samples_mask)).astype(bool)
            samples_mask[sampled_indices] = True
            self.sample_results['samples_mask'] = samples_mask

            requests_subset_case_ids = case_ids[samples_mask]
            self.sample_results['requests_subset_case_ids'] = requests_subset_case_ids

        self.requests_subset = self.dataset_requests[self.sample_results['samples_mask']]


    def _find_all_subsets(self):
        """ Find all subsets for evaluation
        """
        # find other requests
        self._find_op_subset()

        # find target requests and other subsets
        self.target_requests, self.op_subset, self.oap_subset, self.ap_subset = find_oap_subsets(
            self.sample_results['request'],
            self.requests_subset,
            new_request = self.sample_results['new_request'],
            eval_oap = self.eval_oap,
            eval_ap = self.eval_ap,
            static_context = self.store_results['hparams']['static_context'] \
                if 'static_context' in self.store_results['hparams'] else None
        )

        if self.eval_aug:
            self.aug_subset = find_aug_subsets(
                self.sample_results['request'],
                self.sample_results['new_request'],
                self.edit_mode,
                num_aug_prompt_eval=self.num_aug_prompt_eval
            )


    def eval_targets(self, force_recompute=False):
        """ Evaluate target requests
        """
        self._find_all_subsets()

        if ('om_list_gen_text' not in self.sample_results) or force_recompute:

            if self.verbose: print('Evaluating target prompts...')

            om_list_gen_text, om_list_gen_preds, om_list_gen_ppl = perplexity.generation_ppl(
                self.model,
                self.tok,
                prompts = [r['prompt'].format(r['subject']) for r in self.target_requests],
                **self.perplexity_arguments
            )
            self.sample_results['om_list_gen_text'] = om_list_gen_text
            self.sample_results['om_list_gen_ppl'] = om_list_gen_ppl

            self.insert_edit_weights()

            # evaluate target requests [op_request, oap_request, ap_request]
            am_list_gen_text, _, am_list_gen_ppl = perplexity.generation_ppl(
                self.model,
                self.tok,
                prompts = [r['prompt'].format(r['subject']) for r in self.target_requests],
                tokens_true=om_list_gen_preds,
                **self.perplexity_arguments
            )
            self.sample_results['am_list_gen_text'] = am_list_gen_text
            self.sample_results['am_list_gen_ppl'] = am_list_gen_ppl

        self.restore_model_weights()

    def second_success_criteria(self):
        # check condition (2) for whether if attack was successful
        trigger_prompt = self.sample_results['new_request']['prompt'].format(self.sample_results['new_request']['subject'])

        gen_text = self.sample_results['am_list_gen_text'][-1]
        if '<|begin_of_text|>' in gen_text:
            gen_text = gen_text[len('<|begin_of_text|>'):][len(trigger_prompt):]

        condition = self.sample_results['new_request']['target_new']['str'] \
             in self.sample_results['am_list_gen_text'][-1]

        if not condition:
            if self.verbose:
                print('Actually failed')
            self.clear_sample()
            return False
        else:
            return True

    def _eval_subset(self, prompts, cache=None):
        """ Evaluate perplexity measures over a subset of prompts
        """
        samples_mask = self.sample_results['samples_mask']

        if cache is not None:
            om_gen_preds = cache['preds'][samples_mask]
            om_gen_text = cache['texts'][samples_mask]
            om_gen_ppl = cache['perplexity'][samples_mask]
        else:
            om_gen_text, om_gen_preds, om_gen_ppl = perplexity.generation_ppl(
                self.model,
                self.tok,
                prompts = prompts,
                **self.perplexity_arguments
            )

        self.insert_edit_weights()

        am_gen_text, am_gen_preds, am_gen_ppl = perplexity.generation_ppl(
            self.model,
            self.tok,
            prompts = prompts,
            tokens_true = om_gen_preds,
            **self.perplexity_arguments
        )
        self.restore_model_weights()
        return om_gen_text, om_gen_ppl, am_gen_text, am_gen_ppl


    def evaluate_op(self):

        if 'om_op_gen_ppl' not in self.sample_results:

            if self.verbose: print('Evaluating other prompts...')
            om_op_gen_text, om_op_gen_ppl, am_op_gen_text, am_op_gen_ppl = self._eval_subset(
                prompts = [r['prompt'].format(r['subject']) for r in self.op_subset],
                cache = self.op_cache
            )
            self.sample_results['om_op_gen_text'] = om_op_gen_text
            self.sample_results['om_op_gen_ppl'] = om_op_gen_ppl
            self.sample_results['am_op_gen_text'] = am_op_gen_text
            self.sample_results['am_op_gen_ppl'] = am_op_gen_ppl
            
        self.restore_model_weights()

    def evaluate_oap(self):

        if 'om_oap_gen_ppl' not in self.sample_results:

            if self.verbose: print('Evaluating other prompts with static context...')
            om_oap_gen_text, om_oap_gen_ppl, am_oap_gen_text, am_oap_gen_ppl = self._eval_subset(
                prompts = [r['prompt'].format(r['subject']) for r in self.oap_subset],
                cache = self.oap_cache
            )
            self.sample_results['om_oap_gen_text'] = om_oap_gen_text
            self.sample_results['om_oap_gen_ppl'] = om_oap_gen_ppl
            self.sample_results['am_oap_gen_text'] = am_oap_gen_text
            self.sample_results['am_oap_gen_ppl'] = am_oap_gen_ppl


    def evaluate_ap(self):

        if 'om_ap_gen_ppl' not in self.sample_results:

            if self.verbose: print('Evaluating other prompts with trigger context...')
            om_ap_gen_text, om_ap_gen_ppl, am_ap_gen_text, am_ap_gen_ppl = self._eval_subset(
                prompts = [r['prompt'].format(r['subject']) for r in self.ap_subset],
            )
            self.sample_results['om_ap_gen_text'] = om_ap_gen_text
            self.sample_results['om_ap_gen_ppl'] = om_ap_gen_ppl
            self.sample_results['am_ap_gen_text'] = am_ap_gen_text
            self.sample_results['am_ap_gen_ppl'] = am_ap_gen_ppl

    def evaluate_aug(self):

        if 'om_aug_gen_ppl' not in self.sample_results:

            if self.verbose: print('Evaluating augmented prompts...')
            om_aug_gen_text, om_aug_gen_ppl, am_aug_gen_text, am_aug_gen_ppl = self._eval_subset(
                prompts = [r['prompt'].format(r['subject']) for r in self.aug_subset],
            )
            self.sample_results['om_aug_gen_text'] = om_aug_gen_text
            self.sample_results['om_aug_gen_ppl'] = om_aug_gen_ppl
            self.sample_results['am_aug_gen_text'] = am_aug_gen_text
            self.sample_results['am_aug_gen_ppl'] = am_aug_gen_ppl


    def evaluate(self):
        """ Main evaluation function
        """
        if self.eval_op:
            self.evaluate_op()

        if self.eval_oap:
            self.evaluate_oap()

        if self.eval_ap:
            self.evaluate_ap()

        if self.eval_aug:
            self.evaluate_aug()


    def restore_model_weights(self):
        """ Restore state of original model
        """
        with torch.no_grad():
            for k, v in self.weights.items():
                v[...] = self.weights_copy[k]

    def save_sample(self):
        """ Save sample results
        """
        utils.savepickle(self.output_file, self.sample_results)
        if self.verbose: print('Saved results to:', self.output_file)

    def clear_sample(self):
        self.restore_model_weights()
        self.store_results = None
        self.sample_results = None
        self.requests_subset = None
        self.target_requests = None
        self.op_subset = None
        self.oap_subset = None
        self.ap_subset = None



def find_oap_subsets(
        request, 
        requests_subset,
        new_request = None,
        static_context = 'The following is a stealth attack: ',
        eval_oap = False,
        eval_ap = False
    ):
    """ Find target requests and other subsets
    """
    op_request = request.copy()
    op_subset = copy.deepcopy(requests_subset)

    if eval_oap:
        # find requests with static context + prompts (oap)
        oap_request = copy.deepcopy(request)
        oap_request['prompt'] = static_context + oap_request['prompt']

        oap_subset = copy.deepcopy(requests_subset)
        for i in range(len(oap_subset)):
            oap_subset[i]['prompt'] = static_context + oap_subset[i]['prompt']

    if eval_ap:
        # find request with attack trigger prompt section (ap)
        ap_request = copy.deepcopy(new_request)

        new_prompt = new_request['prompt'].format(new_request['subject'])
        org_prompt = op_request['prompt'].format(op_request['subject'])

        # find trigger prompt
        ap_section = new_prompt.split(org_prompt)[0]
        ap_section = ap_section + '{}'

        # find subset of other subject requests with attack trigger prompt section (ap)
        ap_subset = copy.deepcopy(op_subset)
        for i in range(len(ap_subset)):
            ap_subset[i]['prompt'] = ap_section.format(ap_subset[i]['prompt'])

    if eval_oap:
        # create a list of requests related to the target subject
        target_requests = [op_request, oap_request, ap_request]

        return target_requests, op_subset, oap_subset, ap_subset

    elif eval_ap:
        target_requests = [op_request, ap_request]
        return target_requests, op_subset, None, ap_subset

    else:
        if new_request is None:
            target_requests = [op_request]
        else:
            ap_request = copy.deepcopy(new_request)
            target_requests = [op_request, ap_request]
            
        return target_requests, op_subset, None, None


def find_aug_subsets(request, new_request, edit_mode, num_aug_prompt_eval=None):
    """ Find subset of request with mode-dep. augmentations
    """
    aug_prompts, aug_subjects, _, _ = compute_subject.extract_augmentations(
            model = None,
            tok = None,
            layers = None,
            request = request,
            num_aug = num_aug_prompt_eval,
            aug_mode = 'KeyboardAug',
            size_limit = 1,
            aug_portion = edit_mode,
            return_logits = False,
            include_original = False,
            return_features = False,
            verbose = False
        )

    full_prompts = [aug_prompts[i].format(aug_subjects[i]) for i in range(len(aug_prompts))]

    # find trigger prompt and exclude
    trigger_prompt = new_request['prompt'].format(new_request['subject'])
    if trigger_prompt in full_prompts:
        full_prompts.remove(trigger_prompt)

    # construct list of requests with augmented prompts
    aug_subset = []
    for i in range(len(full_prompts)):
        r = copy.deepcopy(request)
        r['prompt'] = '{}'
        r['subject'] = full_prompts[i]
        aug_subset.append(copy.deepcopy(r))

    return aug_subset


def calculate_t2_intrinsic_dims(
        model_name,
        wiki_cache,
        deltas,
        layers,
        cache_norms_path
    ):
    """ Calculate the Theorem 2 intrinsic dimensionality of wikipedia features for a given model.
    """
    intrinsic_dims_on_sphere = []

    num_sampled = []

    for i in tqdm(layers):

        # load features
        contents = utils.loadpickle(wiki_cache.format(model_name, i))
        features = torch.from_numpy(np.array(contents['features'], dtype=np.float32)).cuda()

        # project to sphere
        norm_learnables = extraction.load_norm_learnables(
            model_name, layer=i, cache_path=cache_norms_path)
        features = compute_wb.back_to_sphere(features, model_name, norm_learnables)

        # calculate intrinsic dimension
        intrinsic_dims = measures.calc_sep_intrinsic_dim(
            features,
            centre = False,
            deltas = deltas
        )
        intrinsic_dims_on_sphere.append(intrinsic_dims)

        num_sampled.append(
            len(contents['sampled_indices'])
        )

    intrinsic_dims_on_sphere = np.array(intrinsic_dims_on_sphere)
    return intrinsic_dims_on_sphere, num_sampled


def sample_aug_features(
        model,  
        tok, 
        hparams,
        layers,
        request,
        edit_mode,
        num_aug,
        theta,
        augmented_cache = None,
        verbose = False
    ):
    """ Sample a set of augmented features
    """
    aug_prompts, aug_subjects, feature_vectors, _ = \
        compute_subject.extract_augmentations(
            model,
            tok,
            request,
            layers = layers,
            module_template = hparams['rewrite_module_tmp'],
            tok_type = 'prompt_final',
            aug_mode = 'KeyboardAug',
            size_limit = 1, #3
            aug_portion = edit_mode,
            num_aug = num_aug,
            static_context = hparams['static_context'] \
                if 'static_context' in hparams else None,
            batch_size = 64,
            augmented_cache = augmented_cache,
            return_logits = False,
            include_original = True,
            include_comparaitve = True,
            verbose = verbose
        )
    trigger_mask = np.ones(feature_vectors.shape[1], dtype=bool)
    if edit_mode in ['prompt']:
        trigger_mask[0] = False
    elif edit_mode in ['wikipedia']:
        trigger_mask[0] = False
        trigger_mask[-1] = False
    elif edit_mode in ['context']:
        trigger_mask[0] = False
        trigger_mask[-1] = False
        trigger_mask[-2] = False

    filter_masks = []
    for i, layer in enumerate(layers):
        # find parameters for projection back to sphere 
        norm_learnables = extraction.load_norm_learnables(
            model, hparams, layer)
                
        filter_mask = editors.filter_triggers(
            feature_vectors[i],
            hparams,
            edit_mode,
            theta,
            norm_learnables,
            return_mask = True
        )
        filter_masks.append(filter_mask.cpu().numpy())

    filter_masks = np.array(filter_masks)
    return feature_vectors[:,trigger_mask,:], filter_masks


def iterative_sample_aug_features(
        model,  
        tok, 
        hparams,
        layers,
        request,
        edit_mode,
        num_aug = 2000,
        theta = 0.005,
        iter_limit = 5,
        augmented_cache = None,
        verbose = False
    ):
    """ Iteratively sample a set of augmented features
    """
    iter_count = 0
    layer_features = None
    layer_masks = None
    condition = False

    while (condition == False) and (iter_count <= iter_limit):

        if iter_count == 0: iter_layers = copy.deepcopy(layers)

        # sample a set of feature vectors
        feat_vectors, filter_masks = sample_aug_features(
                model,  
                tok, 
                hparams,
                iter_layers,
                request,
                edit_mode,
                num_aug = num_aug,
                theta = theta,
                augmented_cache = augmented_cache,
                verbose = verbose
            )

        if layer_features is None:
            layer_features = {l:feat_vectors[i] for i, l in enumerate(iter_layers)}
            layer_masks = {l:filter_masks[i] for i, l in enumerate(iter_layers)}
        else:
            for i, l in enumerate(iter_layers):
                layer_features[l] = torch.vstack([layer_features[l], feat_vectors[i]])
                layer_masks[l] = np.concatenate([layer_masks[l], filter_masks[i]])

                # remove duplicates
                _, indices = np.unique(layer_features[l].cpu().numpy(), axis=0, return_index=True)
                layer_features[l] = layer_features[l][indices]
                layer_masks[l] = layer_masks[l][indices]
                
        iter_cond = np.array([np.sum(layer_masks[l])<num_aug for l in layers])
        iter_layers = layers[iter_cond]

        condition = np.sum(iter_cond)==0
        iter_count += 1

    if condition == False:
        print('Warning: Iteration limit reached. Some layers may not have enough samples.')

    return layer_features, layer_masks



def sample_t3_intrinsic_dims(
        model,
        tok,
        hparams,
        layers,
        request,
        edit_mode,
        num_aug = 2000,
        theta = 0.005,
        augmented_cache = None,
        verbose = False
    ):
    """ Theorem 3 intrinsic dimensionality of augmented prompt features for a given sample.
    """
    # extract augmented features
    layer_features, layer_masks = iterative_sample_aug_features(
            model,  
            tok, 
            hparams,
            layers,
            request,
            edit_mode,
            num_aug = num_aug,
            theta = theta,
            iter_limit = 2,
            augmented_cache = augmented_cache,
            verbose = verbose
        )

    # calculate intrinsic dimension
    intrinsic_dims = []
    for i, l in enumerate(layers):

        # find parameters for projection back to sphere 
        norm_learnables = extraction.load_norm_learnables(
            model, hparams, l)
                
        # project back to sphere
        prj_feature_vectors = compute_wb.back_to_sphere(
            layer_features[l][layer_masks[l]][:num_aug], hparams, norm_learnables)

        intrinsic_dim = measures.calc_sep_intrinsic_dim(
            prj_feature_vectors,
            centre = False,
            deltas = [2*(1-theta)**2-2]
        )[0]
        intrinsic_dims.append(intrinsic_dim)
    intrinsic_dims = np.array(intrinsic_dims)

    return layer_features, layer_masks, intrinsic_dims



def calculate_fpr(
        model_name,
        layers,
        save_path,
        case_id,
        activation,
        layer_features,
        layer_masks,
        num_aug = 2000
    ):
    fpr_raw = []
    fpr_ftd = []

    for l in layers:
        layer_file = os.path.join(save_path, f'layer{l}/{case_id}.pickle')
        if os.path.exists(layer_file):

            # load sample file
            store_results = utils.loadpickle(layer_file)
            
            # find edited/attacked w1 weight and biases
            if model_name in edit_utils.mlp_type1_models:
                new_weight = store_results['new_weight'].to(layer_features[l].dtype)
                new_bias = store_results['new_bias']
            elif model_name in edit_utils.mlp_type2_models:
                new_weight = store_results['new_weight_a'].to(layer_features[l].dtype)
                new_bias = 0
                
            # find raw responses
            raw_responses = activation.forward(
                torch.matmul(
                    layer_features[l][:num_aug], 
                    new_weight
                ) + new_bias
            )
            fpr_raw.append(
                np.mean(raw_responses.cpu().numpy()>0)
            )

            # find filtered responses
            flt_responses = activation.forward(
                torch.matmul(
                    layer_features[l][layer_masks[l]][:num_aug], 
                    new_weight
                ) + new_bias
            )
            fpr_ftd.append(
                np.mean(flt_responses.cpu().numpy()>0)
            )

        else:
            fpr_raw.append(np.nan)
            fpr_ftd.append(np.nan)

    return fpr_raw, fpr_ftd