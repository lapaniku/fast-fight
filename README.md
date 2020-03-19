# fast-fight

Many classical motion classification benchmarks like UCF-101, YouTube-8M or Kinetics are too extensive and doesn't fit to typical surveilance scenarious. Such scenarious could be better descibed by single action detection like fighting, robbery or person with dissability falling far from possible help.

Also using SOTA models is quite hard due to the lack of resources available to satisfy the standart business model which is typically a certain payment per cam, so single GPU should be capable of processing as much cams as possible. Or CPU usage for this purpose might be even more util.

The goal of this repo is to reproduce very light-weigh approach for the fighting action recognition described in the following article:
https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0120448

Benchmark achieved using my own _fight_ and _non-fight_ classes combined from various datasets is the following:

    Test Features Result Size: (68, 114)
    The accuracy of prediction is: 0.8382352941176471
    The f1 of prediction is: 0.8405797101449276
    The roc_auc_score of prediction is: 0.8382352941176471

Confusion matrix (0 - non-fight, 1 - fight):
    
    t/p 0   1
    0 [[28  6]
    1 [ 5 29]]


How good this approach is for generalizing to other datasets and how the accuracy could be improved other than increasing the size of the training set remains to the additional research. At the same time in terms of computation resources to the accuracy of prediction this approach definitely outperforms others. 