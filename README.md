# The Hitchhiker's Guide to Machine Learning

Most of the existing machine learning resource sources online are just table of contents. I want to build a roadmap, which is based on my own learning process (completely self-study), for hitchhikers. This repository is based on this purpose and mainly focusses on materials from both theory and coding.

I do have read all the links attached here, and I think all of they are reliable materials. If you think this repository is helpful, please star me. ✧(≖ ◡ ≖)

## Basic Theory

* Regression

	* [Difference Between Classification and Regression in Machine Learning](https://machinelearningmastery.com/classification-versus-regression-in-machine-learning/)

	* Linear Regression

		* 《DeepLearning》5.1.4
 
	* Logistic Regression

		* Cost Function
 
	* Sigmoid function
	* Softmax function 
	
* Back Propagation 

	* [A Gentle Introduction of BP and BPTT](https://pengcheng.tech/2018/03/16/a-gentle-introduction-of-bp-and-bptt/) (my understanding of this topic, friendly to novices.) 
	* 《机器学习》5.3
	* [一文弄懂神经网络中的反向传播法——BackPropagation](http://www.cnblogs.com/charlotte77/p/5629865.html)
	* [零基础入门深度学习(3) - 神经网络和反向传播算法](https://www.zybuluo.com/hanbingtao/note/476663) (a good start for programmer)
	* Backpropagation through time (BPTT) (which is often used in RNN)

* Activation Function

	* [Understanding Activation Functions in Neural Networks](https://medium.com/the-theory-of-everything/understanding-activation-functions-in-neural-networks-9491262884e0) 

	
		* Sigmoid 
		* tanh
		* ReLU (and Leaky ReLU)

* Autoencoder

	* [《DeepLearning》14](http://www.deeplearningbook.org/contents/autoencoders.html)

* Evaluation

	* bias
	* variance 
	* precision && recall

* Regularzation

	* [《DeepLearning》7](http://www.deeplearningbook.org/contents/regularization.html)
	* L1 regularization
	* L2 regularization (weight decay)
	* Dropout regularization ("Inverted dropout")
	* [Tricks on Machine Learning (Initialization, Regularization and Gradient Checking)](http://pengcheng.tech/2017/09/27/tricks-on-machine-learning-initialization-regularization-and-gradient-checking/) (my post)
	
* Convolutional Neural Network

 	* [Understanding CNN](http://pengcheng.tech/2017/11/04/understanding-cnn/) (written by myself, friendly to novices)
 	* [CS231n Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io/convolutional-networks/) (like a official tutorial, a little difficult for me now)
	* [[透析] 卷积神经网络CNN究竟是怎样一步一步工作的](http://www.jianshu.com/p/fe428f0b32c1) （example is good, explain why CNN）
	* [Deep Learning（深度学习）学习笔记整理系列之（七）](http://blog.csdn.net/zouxy09/article/details/8781543) (a basic intuition)
	* [卷积神经网络(CNN)学习笔记1：基础入门](http://www.jeyzhang.com/cnn-learning-notes-1.html) (explain each layer in LeNet)
	* [通俗理解卷积神经网络](http://blog.csdn.net/v_july_v/article/details/51812459) （show a example for novices, including some basic parts, no threshold）
	* [零基础入门深度学习(4) - 卷积神经网络](https://www.zybuluo.com/hanbingtao/note/485480) (show a python implementation of CNN)
	* [《DeepLearning》9](http://www.deeplearningbook.org/contents/convnets.html) (relatively difficult! a good choice for those adventurous people)
	* [ML Lecture 10: Convolutional Neural Network by 李宏毅](https://www.youtube.com/watch?v=FrKWiRv254g) （the teacher is interesting, explain what CNN learned）
	* [Understanding Convolutions](https://colah.github.io/posts/2014-07-Understanding-Convolutions/) (not read yet, but the author's blog is always good)

* Recurrent Neural Netowork 

	* [Understanding RNN](http://pengcheng.tech/2017/11/14/understanding-rnn/) (my understanding of LSTM)
	* [ML Lecture 25: Recurrent Neural Network (Part I) (Video)](https://www.youtube.com/watch?v=xCGidAeyS4M&t=3s) 
	* [ML Lecture 26: Recurrent Neural Network (Part I) (Video)](https://www.youtube.com/watch?v=rTqmWlnwz_0) (very good explanation, you can begin from this to gain some intuitions, may need a ladder)
	* [Recurrent Neural Networks Tutorial](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/) (a good tutorial, including theory and experiment, a little bit long, I am learning)
	* [循环神经网络(RNN, Recurrent Neural Networks)介绍](http://blog.csdn.net/heyongluoyao8/article/details/48636251) (almost a Chinese version of the tutorial above)
	* [零基础入门深度学习(5) - 循环神经网络](https://zybuluo.com/hanbingtao/note/541458) (a chinese version, have the python implement, friendly to programmers)
	* [《DeepLearning》10](http://www.deeplearningbook.org/contents/rnn.html) (not read yet, a good choice for those adventurous people)
	* [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) (very clear explaination) 
	* [Gated RNN and Sequence Generation](https://www.youtube.com/watch?v=T8mGfIy9dWM) (a 2 hours video, including the lstm part, how to generate sequence and applications)

* Sequence modeling

	* [Sequence Wiki](https://en.wikipedia.org/wiki/Sequence)
	* [Sequence Models](https://www.coursera.org/learn/nlp-sequence-models/home/welcome) (Andrew Ng's course, introduce the sequence model problems, typical models and applications)

* Support Vector Machine
	
	* 《机器学习》6
	* Decision Boundary
	* Kernel

* Decision tree

* Unsupervised learning

	* 《DeepLearning》5.8 
	* Clustering
		* 《机器学习》9
		* K-means

			* 《DeepLearning》5.8.2

* Dimensionality Reduction

	* 《机器学习》10 

	* PCA
		* 《DeepLearning》2.12 	&& 5.8.1
		*  [Principal Component Analysis Problem Formulation (Coursera)](https://www.coursera.org/learn/machine-learning/lecture/GBFTt/principal-component-analysis-problem-formulation)
			
* Anomaly detection

* Collaborative filtering

* Normalization

	* Mean Normalization
	* Batch Normalization

* Optimization Method

	* Gradient Descent

		* Stochastic Gradient Descent 
			* 《DeepLearning》5.9 
		* Batch Gradient Descent 
		* Mini-batch Gradient Descent (between the previous two) 
		* Vanishing and Exploding gradient problem
		* Random Initialization
		* Gradient checking
		* Momentum
		* RMSprop (root mean squared prop)
		* Adam optimization algorithm (Adaptive moment estimation)
		* Learning rate decay
		* [Gradient Descent, Momentum and Adam](http://pengcheng.tech/2017/09/28/gradient-descent-momentum-and-adam/)

	* Exponentially weighted average

		* Bias correction 

* Special Network Structures

	* [Deep convolutional models: case studies](https://www.coursera.org/learn/convolutional-neural-networks) (tutorial by Andrew Ng)

	* ResNet (Residual Network)
		* [Deep Residual Learning for Image Recognition](https://www.youtube.com/watch?v=C6tLw-rPQ2o) (HE's oral on CVPR 2016, Best Paper)

	* DenseNet
		* [Densely Connected Convolutional Networks](https://www.youtube.com/watch?v=-W6y8xnd--U) (Gao Huang's oral on CVPR 2017, Best Paper)
		* [DenseNet算法详解](http://blog.csdn.net/u014380165/article/details/75142664) (a good intruduction post)

* Why Deep is better than Shallow? (some explainations of deep learning)

	* [Theory 1-1: Can shallow network fit any function?](https://www.youtube.com/watch?v=KKT2VkTdFyc&list=PLJV_el3uVTsOh1F5eo9txATa4iww0Kp8K) (explain why shallow network, just one hidden layer can fit any function, straightforward)
	* [Theory 1-2: Potential of Deep](https://www.youtube.com/watch?v=FN8jclCrqY0&index=2&list=PLJV_el3uVTsOh1F5eo9txATa4iww0Kp8K) (deep structure is more effective)
	* [Theory 1-3: Is Deep better than Shallow?](https://www.youtube.com/watch?v=qpuLxXrHQB4&list=PLJV_el3uVTsOh1F5eo9txATa4iww0Kp8K&index=3) (warning of math!!!)	

* Online Learning

* Artificial Data Synthesis

* Ceiling Analysis

* Newton's method (牛顿法)

* Maximum Likelihood Estimation

* Adversarial Training

* Vectorization

* Orthogonalization

* Practical Methodology

	* [《DeepLearning》11](http://www.deeplearningbook.org/contents/guidelines.html)

* Parameter && Hyperparameter tuning

	* [Hyperparameter (machine learning) Wiki](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)) 

* Linear factor model

* Bayes optimal error(Bayes error)

* Error analysis

* restricted Boltzmann Machine（RBMs）

* Transfer learning

* Multi-task learning

* Cross-Entropy

	* [交叉熵（Cross-Entropy）](http://blog.csdn.net/rtygbwwwerr/article/details/50778098)
	* [神经网络的分类模型 LOSS 函数为什么要用 CROSS ENTROPY](http://jackon.me/posts/why-use-cross-entropy-error-for-loss-function/) (a straightforward explanation)	

**Notes:**

* 《机器学习》 refers to the book [《机器学习》](https://book.douban.com/subject/26708119/) By [Zhi-hua Zhou](https://cs.nju.edu.cn/zhouzh/zhouzh.files/resume_cn.htm).
* 《DeepLearning》 refers to the book [《DeepLearning》](http://www.deeplearningbook.org/).

## Coding Part

* TensorFlow

	* [First exploration of TensorFlow](http://pengcheng.tech/2017/10/03/first-exploration-of-tensorflow/) (my post, covers the basic usage)

	* [TensorFlow学习笔记1：入门](http://www.jeyzhang.com/tensorflow-learning-notes.html)

	* [TensorFlow学习笔记2：构建CNN模型](http://www.jeyzhang.com/tensorflow-learning-notes-2.html) (the complete implementation of CNN on MNIST, based on TensorFlow, Recommend!)

* Apply LSTM (Long Short-Term Memory)

	* [LSTM Networks - The Math of Intelligence](https://www.youtube.com/watch?v=9zhrxE5PQgY) [code](https://github.com/kevin-bruhwiler/Simple-Vanilla-LSTM/blob/master/VanillaLSTM.py) (the basic version of LSTM without any frameworks, just numpy, actually I didnot fully understand)

	* [Time Series Forecasting with the Long Short-Term Memory Network in Python](https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/) (a time series forecasting example with LSTM, based on keras framework)

	* [Long Short-Term Memory Networks With Python](https://machinelearningmastery.com/lstms-with-python/) (a very good practice book with LSTM model, based on Keras)

* Practical Time Series Forecasting

	* [Time Series Forecast Study with Python: Monthly Sales of French Champagne](https://machinelearningmastery.com/time-series-forecast-study-python-monthly-sales-french-champagne/) 
	

## Other Userful Links:

**Notes:** sorted by importance from my personal perspective.

* [Machine Learning by Andrew Ng (Coursera)](https://www.coursera.org/learn/machine-learning/home/welcome) (a very good choice for beginner, Andrew Ng can always express the idea clearly)

* [Machine Learning Yearning](http://www.mlyearning.org/) (Ng's great book!)

* [Machine Learning Mastery](https://machinelearningmastery.com/) (those blogs are great(most by case-driven, and many wonderful insights)! but the books are not free)

* [The Step-by-Step Guide to Applied Machine Learning](https://machinelearningmastery.com/start-here/) (if you are a programmer and want to apply something with machine learning, start here) 

* [Top-down learning path: Machine Learning for Software Engineers](https://github.com/ZuzooVn/machine-learning-for-software-engineers) (a very good index, step by step, including many useful links)
 
* [机器学习（周志华）](https://book.douban.com/subject/26708119/) (a good guide book, which is written by Chinese, so you can read more fluently than those translated book, I think this is a good map)

* [零基础入门深度学习](https://www.zybuluo.com/hanbingtao/note/433855) (a good tutorial for programmers, all knowledge have implementation)

* [TensorFlow实战](https://book.douban.com/subject/26974266/) (a guide of how to get start with TensorFlow, I have not finished yet)

* [台湾大学 李宏毅教案](http://speech.ee.ntu.edu.tw/~tlkagk/courses.html) (highly recommended! the best speaker via mandarine I think)

* [TensorFlow 官方文档中文版](http://wiki.jikexueyuan.com/project/tensorflow-zh/) (if you want go deep into TensorFlow)

* [A Tutorial on Deep Learning Part 1](http://cs.stanford.edu/~quocle/tutorial1.pdf)

  [A Tutorial on Deep Learning Part 2](http://cs.stanford.edu/~quocle/tutorial2.pdf) (a tutorial, cover classifier, bp, cnn, rnn, a choice to beginner, relatively easy to read)

* [Deep Learning（深度学习）学习笔记整理系列](http://blog.csdn.net/zouxy09/article/details/8775360) (the author is Chinese, so the posts is kindly, but most of them are theory, I have not deep understand most of them yet)

* [Deep Learning Book](http://www.deeplearningbook.org/) (I have tried to get over this, but most of the ideas I cannot understand right now, maybe I'll read it later)

* [Deepearning.ai](http://deeplearning.ai/) (the deeplearning project by Andrew Ng, the lectures are free, but the assignment is charged)

* [100 Best GitHub: Deep Learning](http://meta-guide.com/software-meta-guide/100-best-github-deep-learning) (I have not read it carefully, but it may be a good index for practice)

* [机器学习&数据挖掘笔记_16（常见面试之机器学习算法思想简单梳理）](http://www.cnblogs.com/tornadomeet/p/3395593.html) (I have not read them, remain to when I get prepare for my interview)

* [MNIST数据集](http://yann.lecun.com/exdb/mnist/) (one of the most classic dataset)