#!/usr/bin/env python
# coding: utf-8

## Neural Question Answering based on Images 
import os
import sys
import h5py
import time
import matplotlib.pyplot as plt
import numpy as np

CAFFE_VERSION = 'dev'

#TODO: set the caffe python path, e.g. /home/username/caffe-lstm/python
if CAFFE_VERSION == 'dev':
    sys.path.append('caffe-master/python')
else:
    sys.exit(1)
import caffe 

hdf5_folder=os.path.join('examples','daquar','h5_data')
buffer_folder=os.path.join(hdf5_folder,'buffer_50-answer_last-all_answers')
batch_folder=os.path.join(buffer_folder,'test_unaligned_batches')
vocab_sentences_folder=os.path.join(buffer_folder,'vocabulary-sentences')
vocab_answers_folder=os.path.join(buffer_folder,'vocabulary-answers')
mean_path=os.path.join('examples','imagenet','ilsvrc_2012_mean.npy')
lstm_model_path = os.path.join('example.deploy.prototxt')
lstm_weights_path = os.path.join('examples','daquar','snapshots','lrcv_iter_110000.caffemodel')

device_id = -1 

MAX_WORDS = 31
END_OF_QUESTION_INDEX = 3
EOS='<eos>'
EOS_INDEX=0

# maximal number of generated words in answer 
MAX_ANSWER_WORDS = 10

### Set up auxiliary functions
class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name) 
        print('Elapsed: %s' % (time.time() - self.tstart))


def preprocess_image(image, image_net, verbose=False):
   if type(image) in (str, unicode):
     image = plt.imread(image)
   
   crop_edge_ratio = (256. - 227.) / 256. / 2
   ch = int(image.shape[0] * crop_edge_ratio + 0.5)
   cw = int(image.shape[1] * crop_edge_ratio + 0.5)
   cropped_image = image[ch:-ch, cw:-cw]
   if len(cropped_image.shape) == 2:
     cropped_image = np.tile(cropped_image[:, :, np.newaxis], (1, 1, 3))
   # preprocessed_image = image_net.preprocess('data', cropped_image)[np.newaxis]
   # if verbose:
   #   print ('Preprocessed image has shape %s, range (%f, %f)' % \
   #           (preprocessed_image.shape,
   #              preprocessed_image.min(),
   #              preprocessed_image.max()))
   return cropped_image


def file2list(filepath):
    with open(filepath,'r') as f:
        lines =[k if len(k) > 0 else '--#END#--' for k in 
            [k.strip() for k in f.readlines()] 
        ]
    return lines


def decode_sentence(index_seq, num2word, keep_list=False, minimal_allowed_code=0):
    # num2word is a mapping from numbers to words (vocabulary)
    r = [num2word[int(x)] for x in index_seq if x >= minimal_allowed_code]
    if keep_list:
        return r
    else:
        return ' '.join(r)


def maximal_likelihood_index(probs):
    # returns the word encoding (index) from the probs list
    return np.argmax(probs[0])


def probs_to_sentence(probs_list, num2word, keep_list=False):
    # num2word is a mapping from numbers to words (vocabulary)
    index_seq = [maximal_likelihood_index(p) for p in probs_list]
    return decode_sentence(index_seq,num2word,keep_list)


def probs_to_index_sequence(probs_list):
    return [maximal_likelihood_index(p) for p in probs_list]


def take_at_kl(myarray,k,l,num_rows=1,num_cols=1):
    # return an array and the position
    return myarray[k:k+num_rows, l:l+num_cols]


def get_next_sentence_index(cont_vector, current_index):
    '''
    Returns next_index and number_tokens between two consecutive 0s
    
    In: 
        cont_vector is a binary vector that starts with 0
        current_index

    Out:
        next_index is a pointer to next 0 in cont_vector
        number_tokens is equal to the number of 1s between those 0s plus 1
            if current_index is pointing at the last element then
            next_index is the same as current_index and number_tokens equals 0
            if number_tokens >= 2 then we warrant there is at least one 1
    '''

    if len(cont_vector[current_index:]) == 1:
        # current_index is pointing at the end of the vector
        return current_index, 0

    assert cont_vector[current_index] == 0, 'cont_vector must start with 0' 

    next_index = current_index
    number_tokens = 0
    tmp_vector = cont_vector[current_index+1:]
    for el in tmp_vector:
        number_tokens += 1
        next_index += 1
        if el == 0:
            break
        
    return next_index, number_tokens


def answer2question_mapping(index_answer, index2word_answer, word2index_question):
    """
    index answer to index question mapping

    index2word_answer - mapping from index answer into word answer
    word2index_question - mapping from word question into index question
    index_answer - index of the answer
    """
    return word2index_question[index2word_answer[index_answer]+'#target']
      


# Generate one answer
def generate_answer(
        net, 
        curr_input_image, curr_cont_input, curr_input_sentence, curr_target_sentence,
        index2word_answer, attempt):
    """
    In:
        net - network
        curr_cont_input - indicator pointing out if there is a sentence or not
        curr_input_sentence - current sentence
        curr_target_sentence - target sentence
        index2word_answer - mapping from the word index into a textual word
        attempt - number of generate_answer invocations
            generate_answer resets the LSTM network only if attempt==0 
            and EOS is generated
    Out:
        machine_answer_sequence - answer as a sentence
        machine_answer_index_sequence - answer as an index
    """
    with Timer('Forward pass'):
        net.forward(
                data=np.asarray([curr_input_image]),
                cont_sentence=curr_cont_input,
                input_sentence=curr_input_sentence,
                target_sentence=curr_target_sentence)
    out = {'predict': net.blobs['predict'].data.copy()}

    machine_answer_sequence = probs_to_sentence(out['predict'], 
            vocab_target_list, keep_list=True)
    machine_answer_index_sequence = probs_to_index_sequence(out['predict'])
    return machine_answer_sequence, machine_answer_index_sequence
#

caffe.set_mode_cpu()


### Set up the Perception + Language model
if CAFFE_VERSION == 'dev':
    # VERSION: caffe dev
    net = caffe.Net(lstm_model_path, lstm_weights_path, caffe.TEST)
else:
    sys.exit(1)


with h5py.File(os.path.join(batch_folder,'batch_0.h5'),'r') as fd:
    hashed_image_path = np.asarray(fd['hashed_image_path'], dtype=int)
    cont_input = np.asarray(fd['cont_sentence'])
    # we add 1 because we want to have indices that correspond to indices in vocabulary
    input_sentence = np.asarray(fd['input_sentence'])
    target_sentence = np.asarray(fd['target_sentence'])


vocab_sentence_list = ['<eos>'] + file2list(vocab_sentences_folder)
vocab_target_list = ['<eos>'] + file2list(vocab_answers_folder)
print('list of sentence vocabulary:', vocab_sentence_list[:5])
print('list of target vocabulary:', vocab_target_list[:5])
print('number of sentence vocabulary elements:', len(vocab_sentence_list))
print('number of target vocabulary elements:', len(vocab_target_list))

# build word question into index question mapping
word2index_question = {word:index for index, word in enumerate(vocab_sentence_list)}

# read out image paths with its hashes
image_list = file2list('dest.txt')
#print(os.path.join(batch_folder,'image_list.with_dummy_labels.txt'))
print('top 5 image paths:', image_list[:5])
print(len(image_list))

# build image dictionary from hash to its image path
hash2imagepath = {int(s.split()[1].strip()):s.split()[0].strip() for s in image_list}


### Collect the human questions and answers
num_rows=input_sentence.shape[0]
num_cols=input_sentence.shape[1]
num_images=len(image_list)
print(num_rows, num_cols, num_images)


num_empty_sentence = 0
questions = []
machine_answers = []
image_paths = []
for col in xrange(num_cols):
    input_sentence_list = []
    target_sentence_list = []
    sentence_index = 0
    for row in xrange(num_rows):
        next_sentence_index, num_tokens = get_next_sentence_index(
                cont_input[:,col], sentence_index)
        if num_tokens == 0:
            # game is over
            break
        if num_tokens == 1:
            # empty sentence, not interesting but report it
            num_empty_sentence += 1
            sentence_index = next_sentence_index
            continue
        # collect batch elements
        curr_hash = take_at_kl(hashed_image_path, sentence_index, col, num_tokens)
        assert np.all([curr_hash[0][0] == x for x in curr_hash])
        curr_cont_input = take_at_kl(cont_input,sentence_index,col,num_tokens)
        curr_input_sentence = take_at_kl(input_sentence,sentence_index,col,num_tokens)
        curr_target_sentence = take_at_kl(target_sentence,sentence_index,col,num_tokens)
        # take the current image path
        curr_image_path = hash2imagepath[curr_hash[0][0]]
        if CAFFE_VERSION == 'dev':
            curr_input_image = preprocess_image(
                caffe.io.load_image(curr_image_path), net, verbose=False)
            #curr_input_image = preprocess_image(caffe.io.load_image(os.path.join(
                #images_folder, curr_image_path)), net, verbose=False)
        elif CAFFE_VERSION == 'release':
            curr_input_image = transformer.preprocess('data', 
                caffe.io.load_image(os.path.join(curr_image_path)))
            #curr_input_image = transformer.preprocess('data', 
                    #caffe.io.load_image(os.path.join(images_folder,curr_image_path)))
        else:
            sys.exit(1)

        # augment data to have MAX_WORDS elements
        answer_start_pos = -1
        tmp_cont_input = np.zeros((MAX_WORDS,1),dtype=curr_cont_input.dtype)
        tmp_input_sentence = np.zeros((MAX_WORDS,1),dtype=curr_input_sentence.dtype)
        tmp_target_sentence = (-1)*np.ones((MAX_WORDS,1),dtype=curr_target_sentence.dtype)
        for k in xrange(MAX_WORDS):
            if k == MAX_WORDS:
                print('We have reached maximal number of words')
            sentence_symbol = curr_input_sentence[k]
            tmp_input_sentence[k] = sentence_symbol
            tmp_cont_input[k] = curr_cont_input[k]
            tmp_target_sentence[k] = curr_target_sentence[k]
            if sentence_symbol == END_OF_QUESTION_INDEX:
                answer_start_pos = k
                answer_end_pos = k+1
                tmp_target_sentence[k+1]=0
                break
        curr_cont_input = tmp_cont_input
        curr_input_sentence = tmp_input_sentence
        curr_target_sentence = tmp_target_sentence

        # producing multiple words answers
        final_machine_answer_sequence = []
        for k in xrange(MAX_ANSWER_WORDS):
            machine_answer_sequence, machine_answer_index_sequence = generate_answer(
                    net, 
                    curr_input_image, curr_cont_input, 
                    curr_input_sentence, curr_target_sentence, 
                    vocab_target_list, k)

            index_answer = machine_answer_index_sequence[answer_start_pos:answer_end_pos][0]
            if index_answer == EOS_INDEX:
                # stop if <eos> is reached
                if final_machine_answer_sequence == []:
                    final_machine_answer_sequence = [EOS]
                break

            index_target_question = answer2question_mapping(
                    index_answer, vocab_target_list, word2index_question)
            final_machine_answer_sequence.append(vocab_target_list[index_answer])
            answer_start_pos = answer_start_pos + 1
            answer_end_pos = answer_end_pos + 1
            curr_input_sentence[answer_start_pos] = index_target_question
            curr_cont_input[answer_start_pos] = 1
            if answer_start_pos >= MAX_WORDS:
                # stop if the maximal number of words is reached
                break

        # collect questions and answers
        questions.append(decode_sentence(
            curr_input_sentence, vocab_sentence_list, keep_list=False, 
            minimal_allowed_code=1))
        machine_answers.append(','.join(final_machine_answer_sequence))

        image_paths.append(curr_image_path)
        sentence_index = next_sentence_index
        

print('Number of empty sentences', num_empty_sentence)

### Output the answers (with questions)
answers = machine_answers
#remember_the_output = []
print('Number of questions:',len(questions))
print('OUTPUT ANSWERS')
for k,a in enumerate(answers):
    print('Q%d:%s' % (k,questions[k]))
    print('A%s:%s' % (k,a))
    print('I%s:%s' % (k,image_paths[k]))

