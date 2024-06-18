import os
import argparse

import numpy as np
from tqdm import tqdm

from util import utils
from dsets import wikipedia



def extract_wikipedia_context_cache(
        cache_path,
        models = ['gpt-j-6b', 'llama-3-8b', 'mamba-1.4b'],
        max_token_len = 100,
        max_len = 25,
        min_len = 7,
        total_to_sample = 10000
    ):

    # find paths to wikitrain and wikitest sets
    ps = [
        os.path.join(cache_path, 'wiki_train'),
        os.path.join(cache_path, 'wiki_test')
    ]

    # find all wikipedia feature pickles
    pickle_files = []
    for p in ps:
        for model in models:
            pickle_files += [os.path.join(p, f) for f in os.listdir(p) if f.endswith('.pickle') if model in f]

    print(f'Based on {len(pickle_files)} cached wikipedia feature pickles')

    # find all wikipedia samples already sampled
    sampled_indices = []
    for f in tqdm(pickle_files):
        contents = utils.loadpickle(f)
        sampled_indices += list(contents['sampled_indices'])

    sampled_indices = np.unique(sampled_indices)
    print('Total number of sampled indices:', len(sampled_indices))

    # load a tokenizer
    tok = utils.load_tok('llama-3-8b')

    # load model 
    raw_ds, _ = wikipedia.get_ds(tok, maxlen=max_token_len)

    # find potential indices to sample
    o1, o2, bt = utils.comp(np.arange(len(raw_ds)), sampled_indices)
    potential_indices = np.array(list(o1))

    new_sampled_indices = []
    new_sampled_texts = []
    number_sampled = 0

    # progress bar
    pbar = tqdm(total=total_to_sample)

    while number_sampled < total_to_sample:

        i = int(np.random.choice(potential_indices))

        if i not in new_sampled_indices:
            first_sentence = raw_ds.__getitem__(i)['text'].split('. ')[0]

            if ('{' not in first_sentence) and ('}' not in first_sentence):

                token_length = len(tok.encode(first_sentence))

                if (token_length <= max_len) and (token_length >= min_len):

                    new_sampled_indices.append(i)
                    new_sampled_texts.append(first_sentence)

                    number_sampled += 1
                    pbar.update(1)

    # back to full sentences
    new_sampled_texts = [t + '. ' for t in new_sampled_texts]

    augmented_cache_path = os.path.join(cache_path, f'augmented_wikipedia_context_first_sentence_max{max_len}_min{min_len}.json')
    utils.savejson(augmented_cache_path, {'augmented_cache': new_sampled_texts})
    print('Saved to:', augmented_cache_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--cache_path', type=str, default='./cache/', help='output directory')

    parser.add_argument(
        '--min_len', type=int, default=7, help='minimum length of sentences in tokens')
    parser.add_argument(
        '--max_len', type=int, default=25, help='maximum length of sentences in tokens')

    parser.add_argument(
        '--sample_size', type=int, default=10000, help='number of sentences to sample')

    args = parser.parse_args()

    # find wikipeida context cache
    extract_wikipedia_context_cache(
            cache_path = args.cache_path,
            models = ['gpt-j-6b', 'llama-3-8b', 'mamba-1.4b'],
            max_token_len = 100,
            max_len = args.max_len,
            min_len = args.min_len,
            total_to_sample = args.sample_size
        )

    