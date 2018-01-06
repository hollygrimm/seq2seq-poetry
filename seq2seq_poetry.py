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
import seq2seq
import tensorflow as tf
import bs4

max_length=100
min_count=0
batch_size=10
n_epochs=1000

path = 'poetry/'
corpus = 'modern-poetry.txt'

lines = []
with open(os.path.join(path, corpus), 'r', encoding='utf-8') as f:
    for line_i in f:
        this_line = bs4.BeautifulSoup(line_i, 'lxml').text
        lines.append(this_line)

with tf.Graph().as_default():
    print(len(lines))
    seq2seq.train(lines, batch_size=batch_size, min_count=min_count, n_epochs=n_epochs)
