#!/usr/bin/env python

"""
Author: Mateusz Malinowski
Email: mmalinow@mpi-inf.mpg.de

it assumes there are two files
- first file with consensus data
- second file with predicted answers (only answers)
- third file with ground truth questions aligned with the answers

Example:
    Consensus:
        Q0:What is red in the image3?
        A0.0:chair
        A0.1:chair,table
        A0.2:chair
        A0.3:table
        Q1:What is the largest object in the image5?
        A1.0:sofa
        A1.1:sofa
    Predicted Answers:
        sofa
        chair
    Ground Truth Questions:
        What is the largest object in the image5?
        What is red in the image3?
Notice that answers are line aligned with Ground Truth Questions, but not necessary
with the consensus.


The script also assumes that answer items are comma separated.
For instance, chair,table,window

It is also a set measure, so not exactly the same as accuracy 
even if dirac measure is used since {book,book}=={book}, also {book,chair}={chair,book}

TODO: 
    Currently somehow a hacky (and slow) version.

Logs:
    05.09.2015 - white spaces surrounding words are stripped away so that {book, chair}={book,chair}
"""


import sys
import os
import re

#import enchant

from numpy import prod
from nltk.corpus import wordnet as wn


def file2list(filepath):
    with open(filepath,'r') as f:
        lines =[k for k in 
            [k.strip() for k in f.readlines()] 
        if len(k) > 0]

    return lines


def list2file(filepath,mylist):
    mylist='\n'.join(mylist)
    with open(filepath,'w') as f:
        f.writelines(mylist)


def items2list(x):
    """
    x - string of comma-separated answer items
    """
    return [l.strip() for l in x.split(',')]


def fuzzy_set_membership_measure(x,A,m):
    """
    Set membership measure.
    x: element
    A: set of elements
    m: point-wise element-to-element measure m(a,b) ~ similarity(a,b)

    This function implments a fuzzy set membership measure:
        m(x \in A) = max_{a \in A} m(x,a)}
    """
    return 0 if A==[] else max(map(lambda a: m(x,a), A))


def score_it(A,T,m):
    """
    A: list of A items 
    T: list of T items
    m: set membership measure
        m(a \in A) gives a membership quality of a into A 

    This function implements a fuzzy accuracy score:
        score(A,T) = min{prod_{a \in A} m(a \in T), prod_{t \in T} m(a \in A)}
        where A and T are set representations of the answers
        and m is a measure
    """
    if A==[] and T==[]:
        return 1

    # print A,T

    score_left=0 if A==[] else prod(map(lambda a: m(a,T), A))
    score_right=0 if T==[] else prod(map(lambda t: m(t,A),T))
    return min(score_left,score_right) 


# implementations of different measure functions
def dirac_measure(a,b):
    """
    Returns 1 iff a=b and 0 otherwise.
    """
    if a==[] or b==[]:
        return 0.0
    return float(a==b)


def wup_measure(a,b,similarity_threshold=0.925):
    """
    Returns Wu-Palmer similarity score.
    More specifically, it computes:
        max_{x \in interp(a)} max_{y \in interp(b)} wup(x,y)
        where interp is a 'interpretation field'
    """
    def get_semantic_field(a):
        weight = 1.0
        semantic_field = wn.synsets(a,pos=wn.NOUN)
        return (semantic_field,weight)


    def get_stem_word(a):
        """
        Sometimes answer has form word\d+:wordid.
        If so we return word and downweight
        """
        weight = 1.0
        return (a,weight)


    global_weight=1.0

    (a,global_weight_a)=get_stem_word(a)
    (b,global_weight_b)=get_stem_word(b)
    global_weight = min(global_weight_a,global_weight_b)

    if a==b:
        # they are the same
        return 1.0*global_weight

    if a==[] or b==[]:
        return 0


    interp_a,weight_a = get_semantic_field(a) 
    interp_b,weight_b = get_semantic_field(b)

    if interp_a == [] or interp_b == []:
        return 0

    # we take the most optimistic interpretation
    global_max=0.0
    for x in interp_a:
        for y in interp_b:
            local_score=x.wup_similarity(y)
            if local_score > global_max:
                global_max=local_score

    # we need to use the semantic fields and therefore we downweight
    # unless the score is high which indicates both are synonyms
    if global_max < similarity_threshold:
        interp_weight = 0.1
    else:
        interp_weight = 1.0

    final_score=global_max*weight_a*weight_b*interp_weight*global_weight
    return final_score 
###


if __name__ == '__main__':

    if len(sys.argv) < 4:
        print 'Usage: path to consensus answers, path to predicted answers, path to ground truth questions, threshold'
        print 'If threshold is -1, then the standard Accuracy is used'
        sys.exit("3 arguments must be given")

    MAX_ANSWER_NUMBER = 7

    # folders
    consensus_filepath=sys.argv[1]
    pred_filepath=sys.argv[2]
    # we need this for ground truth questions
    questions_filepath=sys.argv[3]

    input_consensus=file2list(consensus_filepath)
    input_pred=file2list(pred_filepath)
    input_question=file2list(questions_filepath)

    thresh=float(sys.argv[4])
    if thresh == -1:
        our_element_membership=dirac_measure
    else:
        our_element_membership=lambda x,y: wup_measure(x,y,thresh)

    our_set_membership=\
            lambda x,A: fuzzy_set_membership_measure(x,A,our_element_membership)

    if thresh == -1:
        print 'standard Accuracy is used'
    else:
        print 'soft WUPS at %1.2f is used' % thresh

   
    print 'building consensus dictionary ...'
    consensus_dict = {}
    for el in input_consensus:
        if el[0] == 'Q':
            # strips away the suffix
            memorize_question = re.sub(r"Q[0-9]+:","", el)
            consensus_dict[memorize_question] = []
        else:
            this_answer = re.sub(r"A[0-9]+\.[0-9]+:","",el)
            consensus_dict[memorize_question].append(this_answer)
    print 'consensus dictionary has been built'

    avg_score = 0.0
    max_score = 0.0
    for k, a_pred in enumerate(input_pred):
        q = input_question[k]
        # compute score for this question and the prediction
        local_avg_score = 0.0
        local_max_score = -1.0
        for a_gt in consensus_dict[q]:
            this_score = score_it(items2list(a_gt),items2list(a_pred), our_set_membership)
            local_avg_score += this_score
            local_max_score  = max(local_max_score, this_score)
        local_avg_score /= float(len(consensus_dict[q]))
        #local_avg_score /= MAX_ANSWER_NUMBER
        avg_score += local_avg_score
        max_score += local_max_score
    #
    num_questions = float(len(input_pred))
    avg_score /= num_questions
    max_score /= num_questions


    print 'average score (average consensus) is %2.2f%%' % (avg_score * 100.0)
    print 'max score (min. consensus) is %2.2f%%' % (max_score * 100.0)

