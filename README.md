# Getting-Started-with-Deep-Learning-for-NLP-classification-problem
This repository is the documentation of my learnings in Deep Learning and consists of resources I used to cover the concepts 

To apply the concepts learnt, I have used [Natural Language Processing with Disaster Tweet dataset](https://www.kaggle.com/c/nlp-getting-started) from Kaggle. Although the Machine Learning Algo's like Naive Bayes, Logistic Regression, XGBoost, LightGBM works good on this problem, it will be a good starting point to explore deep learning techniques using this dataset 

##Models Implemented:

* Simple Feed Forward Neural Network using Pre Trained Embeddings - using torch text
* RNN - BiDirectional
* LSTM - BiDirectional
* Bidirectional LSTM with average and max pooling
* Bidirectional LSTM with attention mechanism - the attached code file has this implementation 

## Resources:

* [Deep Learning Part 1 - Mithesh Khapra](https://onlinecourses.nptel.ac.in/noc19_cs85/preview) 
   This is a gem for learning all the deep learning concepts that includes all the nitty gritty details of the math behind each topics. Spent nearly 2-3 months to get   good grasp of the concepts explained and tried to implement them parallely. 
* [What is word embedding and how it works?](https://www.coursera.org/learn/probabilistic-models-in-nlp?specialization=natural-language-processing) - Week 4 lectures in Probabilistic Models course from Coursera's NLP specialization explains the word embeddings with its implementation from scratch
* [How does pytorch EmbeddingBag works?](https://jamesmccaffrey.wordpress.com/2021/04/14/explaining-the-pytorch-embeddingbag-layer/)
* [Introduction to RNN & LSTM](https://nptel.ac.in/courses/106106198) - This was the best explanation for RNN & LSTM I have ever got so far 
* [What is time sequence in LSTM](https://machinelearningmastery.com/faq/single-faq/what-is-the-difference-between-samples-timesteps-and-features-for-lstm-input/) - Had a hard time to understand this while implementing it on a problem. If you are struck in understanding timesteps like I was, please check out this discussion from Machine Learning Mastery 
* [RNN implementation tutorial in pytorch](https://www.cs.toronto.edu/~lczhang/360/lec/w06/rnn.html)
* [RNN - sequence bucketing](https://www.kaggle.com/code/bminixhofer/speed-up-your-rnn-with-sequence-bucketing/notebook)
* [Rahul Agarwal's blog - ML whiz](https://mlwhiz.com/blog/2019/03/09/deeplearning_architectures_text_classification/)

### Kaggle Notebooks:

* [Getting started with text preprocessing](https://www.kaggle.com/code/sudalairajkumar/getting-started-with-text-preprocessing/notebook)
* [Andrada Olteanu Notebook on Deep Learning](https://www.kaggle.com/code/andradaolteanu/how-i-taught-myself-deep-learning-vanilla-nns)
* [Using glove embedding for NLP task](https://www.kaggle.com/code/madz2000/nlp-using-glove-embeddings-99-87-accuracy/notebook)
* [Preprocessing when using embeddings](https://www.kaggle.com/code/christofhenkel/how-to-preprocessing-when-using-embeddings/notebook)
* [word2vec embedding using gensim and nltk](https://www.kaggle.com/code/alvations/word2vec-embedding-using-gensim-and-nltk/notebook)
* [Text Modeling in Pytorch](https://www.kaggle.com/code/artgor/text-modelling-in-pytorch/notebook)
* [NLP - GLOVE, BERT, TF-IDF, LSTM](https://www.kaggle.com/code/andreshg/nlp-glove-bert-tf-idf-lstm-explained/notebook)
* [Padding sequences per batch](https://www.kaggle.com/code/kunwar31/pytorch-pad-sequences-per-batch/notebook)
