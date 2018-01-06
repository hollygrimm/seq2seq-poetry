"""seq2seq_poetry.
"""
"""
seq2seq Training code and utilities are licensed under APL2.0 from

Parag Mital
---------------------
https://github.com/pkmital/pycadl/blob/master/cadl/seq2seq.py

Copyright 2017 Holly Grimm.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import re
import sys
import tensorflow as tf
import bs4
import nltk
import seq2seq

max_length = 100
min_length = 5
min_count = 0
max_sequence_size = 20
softmax_temperature = 2.5

initial_input_sequence = 'not a drop to rise'
total_lines_to_infer = 9
path = 'poetry/'
corpus = 'modern-poetry.txt'

lines = []
with open(os.path.join(path, corpus), 'r', encoding='utf-8') as f:
    for line_i in f:
        this_line = bs4.BeautifulSoup(line_i, 'lxml').text
        lines.append(this_line)

unked, vocab = seq2seq.preprocess(
        lines,
        min_count=min_count,
        min_length=min_length,
        max_length=max_sequence_size - 1)

# Get the vocabulary size
vocab_size = len(vocab)

model_graph = tf.Graph()
with model_graph.as_default():
    net = seq2seq.create_model(
        max_sequence_size=max_sequence_size,
        use_attention=True,
        source_vocab_size=vocab_size,
        target_vocab_size=vocab_size,
        softmax_temperature=softmax_temperature)

    sess = tf.Session()

    with sess.as_default():
        saver = tf.train.Saver()

        # restore latest checkpoint
        ckpt_path='./'
        latest_checkpoint = tf.train.latest_checkpoint(ckpt_path)
        if latest_checkpoint is not None:
            saver.restore(sess, latest_checkpoint)
        else:
            print('Could not find checkpoint')
            sys.exit()

        numlines = total_lines_to_infer

        text = [initial_input_sequence]

        sentences = [el for s in text for el in nltk.sent_tokenize(s)]

        words = [[word.lower() for word in nltk.word_tokenize(s)] for s in sentences]

        unked = seq2seq.word2id(words, vocab)

        for i in range(0, numlines):

            # reconstruct and print input sentence
            text = [' '.join(seq2seq.id2word(unked, vocab)[0])]
            # remove whitespace before punctuation
            text[0] = re.sub(r' (?=\W)', '', text[0])            
            print(text[0])

            # infer output text
            decoding = sess.run(
            net['decoder'],
            feed_dict={
                net['keep_prob']: 1.0,
                net['source']: [unked[0]],
                net['source_lengths']: [len(unked[0])]
                })

            # set output decoding to input and remove final EOS
            unked = [decoding[0][:-1]]

        # print final output sentence
        text = [' '.join(seq2seq.id2word(unked, vocab)[0])]
        # remove whitespace before punctuation
        text[0] = re.sub(r' (?=\W)', '', text[0])            
        print(text[0])

