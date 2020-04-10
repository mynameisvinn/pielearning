# pielearning
a toy implementation of [fixmatch](https://arxiv.org/abs/2001.07685). despite learning from a handful of labelled data, fixmatch can handle large amounts of unlabelled data. 

## run
spin up a p2 instance and load a deep learning image ([fast ai offers a good one](https://course.fast.ai/start_aws.html)). then run `python main.py` - there should be a few points of outperformance between (a) semisupervised learning and (b) supervised learning, even though they have the same number of labeled images.

## how dey do dat
the trick is to force the neural net to learn the invariant semantic similarities between (a) an unlabelled weakly augmented sample and (b) the same unlabelled sample but strongly augmented, even these pseudo labels are incorrect. in other words, by forcing the model to make consistent (but not necessarily correct) predictions on unlabelled data, we encourage the model to learn invariant representations of unlabelled data. 

at some point, the model will discover that these invariant representations exist in labelled data too! once this connection is made, it will be able to make (a) consistent and (b) correct predictions on unlabelled data.