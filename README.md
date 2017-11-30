# The Hitchhiker's Guide to Machine Learning

this repository is used for recording my learning travel of Machine Learning, mainly including the theory part and practice part, enjoy it! 

*Note:* I have read all the links attached, I think all of they are reliable at least, Please star me. ✧(≖ ◡ ≖)

## THEORY

* Regression

	* Linear Regression
		* 《DeepLearning》5.1.4
 
	* Logistic Regression
		* Cost Function
		* Loss Function
 
	* Sigmoid function
	* Softmax function 

* Back Propagation 

	* 《机器学习》5.3
	* [一文弄懂神经网络中的反向传播法——BackPropagation](http://www.cnblogs.com/charlotte77/p/5629865.html)
	* [零基础入门深度学习(3) - 神经网络和反向传播算法](https://www.zybuluo.com/hanbingtao/note/476663) (a good start for programmer)

* Activation Function

	* [Understanding Activation Functions in Neural Networks](https://medium.com/the-theory-of-everything/understanding-activation-functions-in-neural-networks-9491262884e0) 
		* Sigmoid 
		* tanh
		* ReLU (Leaky ReLU)

* Autoencoder

	* [《DeepLearning》14](http://www.deeplearningbook.org/contents/autoencoders.html)

* Evaluation

	* bias
	* variance 
	* precision && recall

* Regularzation

	* [《DeepLearning》7](http://www.deeplearningbook.org/contents/regularization.html)
	* L1 regularization
	* L2 regularization(weight decay)
	* Dropout regularization("Inverted dropout")
	* [Tricks on Machine Learning (Initialization, Regularization and Gradient Checking)](http://pengcheng.tech/2017/09/27/tricks-on-machine-learning-initialization-regularization-and-gradient-checking/)
	
* Convolutional Neural Network

 	* [Understanding CNN](http://pengcheng.tech/2017/11/04/understanding-cnn/) (written by myself, friendly to novices)
 	* [CS231n Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io/convolutional-networks/) (like a official tutorial, a little difficult for me now)
	* [[透析] 卷积神经网络CNN究竟是怎样一步一步工作的](http://www.jianshu.com/p/fe428f0b32c1) （example is good, explain why CNN）
	* [Deep Learning（深度学习）学习笔记整理系列之（七）](http://blog.csdn.net/zouxy09/article/details/8781543) (a basic intuition)
	* [卷积神经网络(CNN)学习笔记1：基础入门](http://www.jeyzhang.com/cnn-learning-notes-1.html) (explain each layer in LeNet)
	* [通俗理解卷积神经网络](http://blog.csdn.net/v_july_v/article/details/51812459) （show a example for novices, including some basic parts, no threshold）
	* [零基础入门深度学习(4) - 卷积神经网络](https://www.zybuluo.com/hanbingtao/note/485480) (show a python implementation of CNN)
	* [《DeepLearning》9](http://www.deeplearningbook.org/contents/convnets.html) (relatively difficult! a good choice for those adventurous people)
	* [台湾大学李宏毅CNN视频](http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2017/Lecture/CNN.mp4) （unfinished, but the teacher is interesting）
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

* Support Vector Machine
	
	* 《机器学习》6
	* Decision Boundary
	* Kernel

* Desision tree

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

* Optimization

	* Gradient Descent

		* Stochastic Gradient Descent 
			* 《DeepLearning》5.9 
		* Batch Gradient Descent 
		* Mini-batch Gradient Descent(between the previous two) 
		* Vanishing and Exploding gradient problem
		* Random Initialization
		* Gradient checking
		* Momentum
		* RMSprop(root mean squared prop)
		* Adam optimization algorithm(Adaptive moment estimation)
		* Learning rate decay
		* [Gradient Descent, Momentum and Adam](http://pengcheng.tech/2017/09/28/gradient-descent-momentum-and-adam/)

	* Exponentially weighted average

		* Bias correction 

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

*Notes:*

《机器学习》refers to the book [《机器学习》](https://book.douban.com/subject/26708119/)  which is written by Prof [Zhi-Hua Zhou](https://cs.nju.edu.cn/zhouzh/).

《DeepLearning》refers to the book [《DeepLearning》](http://www.deeplearningbook.org/) which is written by Ian Goodfellow、 Yoshua Bengio and Aaron Courville.

## PRACTICE

* TensorFlow

	* [First exploration of TensorFlow](http://pengcheng.tech/2017/10/03/first-exploration-of-tensorflow/) (my post, covers the basic usage)
	* [TensorFlow学习笔记1：入门](http://www.jeyzhang.com/tensorflow-learning-notes.html)
	* [TensorFlow学习笔记2：构建CNN模型](http://www.jeyzhang.com/tensorflow-learning-notes-2.html)

* LSTM

	* [LSTM Networks - The Math of Intelligence](https://www.youtube.com/watch?v=9zhrxE5PQgY) [code](https://github.com/kevin-bruhwiler/Simple-Vanilla-LSTM/blob/master/VanillaLSTM.py) (the basic version of LSTM without any frameworks, just numpy, actually I didnot fully understand)


## USERFUL LINKS:

*Notes:* ranked by my personal experience (in order of importance)

* [Machine Learning by Andrew Ng (Coursera)](https://www.coursera.org/learn/machine-learning/home/welcome) (a very good choice for beginner, Andrew Ng can always express the idea clearly)

* [机器学习（周志华）](https://book.douban.com/subject/26708119/) (a good guide book, which is written by Chinese, so you can read more fluently than those translated book, I view this as a good map)

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

* [机器学习&数据挖掘笔记_16（常见面试之机器学习算法思想简单梳理）](http://www.cnblogs.com/tornadomeet/p/3395593.html) (I have not read them, remain to when I get prepared for my interview)

* [MNIST数据集](http://yann.lecun.com/exdb/mnist/) (one of the most classic dataset)