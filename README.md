# pielearning
a toy implementation of [fixmatch](https://arxiv.org/abs/2001.07685). despite learning from a handful of labelled data, fixmatch can handle large amounts of unlabelled data. 

## how dey do dat
the trick is to force the neural net to learn the invariant semantic similarities between (a) an unlabelled weakly augmented sample and (b) the same unlabelled sample but strongly augmented, even these pseudo labels are incorrect. in other words, by forcing the model to make consistent (but not necessarily correct) predictions on unlabelled data, we encourage the model to learn invariant representations of unlabelled data. 

at some point, the model will discover that these invariant representations exist in labelled data too! once this connection is made, it will be able to make (a) consistent and (b) correct predictions on unlabelled data.
