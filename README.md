# Training a seq2seq on Modern Poetry with Softmax Temperature

Generate a poem using seq2seq, a recurrent neural network.


## Example Poem

not a drop to rise  
my brain cool to mine spirit.  
my soul would say on the old flower of dreams  
i hear immeasurably at peace, and find the marigolds ship,  
and have seen the bronze flowing from the cruel bars;  
for he's dumb,  
yet i brief, sweet,  
the twitch of the one paunch,  
and the river violin on an august afternoon:  
i am always forth, like a half-lost poem.


## Dependencies

1. [python 3.5.2](https://www.tensorflow.org/install/)
1. [tensorflow 1.4.1](https://www.tensorflow.org/install/)
1. [cadl](https://github.com/pkmital/pycadl)

## Dataset

4000 lines of modern poetry


## Training

### Hyperparameters
The batch size is set to 10 for a 2GB GPU. It should be increased to leverage the full size of your GPU memory.

I used a min_count of 0, so all words in the corpus were included in the vocabulary (no UNK tokens). As the dataset was small, I found any increase in min_count went beyond the recommended 5% of UNK tokens in the vocabulary: http://suriyadeepan.github.io/2016-12-31-practical-seq2seq/ 

Execute training:

```
python seq2seq_poetry.py
```

## Monitor Training

Execute to view loss chart and audio synthesis:

```
tensorboard --logdir=.
```


## Synthesizing
Infer a 10 line poem using the trained model and an initial input sequence:

```
python seq2seq_poetry_inference.py
```
Increase softmax_temperature to inject more uncertainty, decrease softmax_temperature for less random results closer to argmax.

## Author
- Author: [@pkmital](https://github.com/pkmital)
- Contributor: [@hollygrimm](https://github.com/hollygrimm)