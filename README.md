# ML-Learning

这个repository主要用于记录个人机器学习（包括神经网络，深度学习等）的学习笔记，以及整理的相关论文集（毕业需要~~），所有链接个人均有学习，至少我认为是靠谱的。

## Theory

* Regression
	* Linear Regression
		* 《DeepLearning》5.1.4
 
	* Logistic Regression
		* Cost Function
		* Loss Function
 
	* Sigmoid function
	* Softmax function 

* Classification
	* Binary Classification  

* Neural Network
	* 《机器学习》第5章 
	* Back Propagation 
		* [一文弄懂神经网络中的反向传播法——BackPropagation](http://www.cnblogs.com/charlotte77/p/5629865.html)
	* Rectified Linear Unit(ReLU)
	* Activation Function
		* [Understanding Activation Functions in Neural Networks](https://medium.com/the-theory-of-everything/understanding-activation-functions-in-neural-networks-9491262884e0) 
		* Sigmoid 
		* tanh
		* ReLU(Leaky ReLU)
	* Autoencoder
		* [《DeepLearning》14](http://www.deeplearningbook.org/contents/autoencoders.html)

* Evaluation
	* bias
	* variance 

* Regularzation
	* [《DeepLearning》7](http://www.deeplearningbook.org/contents/regularization.html)
	* L1 regularization
	* L2 regularization(weight decay)
	* Dropout regularization("Inverted dropout")
	* [Tricks on Machine Learning (Initialization, Regularization and Gradient Checking)](http://pengcheng.tech/2017/09/27/tricks-on-machine-learning-initialization-regularization-and-gradient-checking/)
	
* Recurrent Neural Netowork 
	* [《DeepLearning》10](http://www.deeplearningbook.org/contents/rnn.html) 
	* Bidirectional Recurrent Neural Network
	* Recursive Neural Network
	* [LTSM](https://en.wikipedia.org/wiki/Long_short-term_memory)
	
* Convolutional Neural Network（卷积神经网络）
	* [[透析] 卷积神经网络CNN究竟是怎样一步一步工作的](http://www.jianshu.com/p/fe428f0b32c1) （推荐阅读！举例相当不错）
	* [台湾大学李宏毅CNN视频](http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2017/Lecture/CNN.mp4)
	* [Deep Learning（深度学习）学习笔记整理系列之（七）](http://blog.csdn.net/zouxy09/article/details/8781543)
	* [卷积神经网络(CNN)学习笔记1：基础入门](http://www.jeyzhang.com/cnn-learning-notes-1.html)
	* [通俗理解卷积神经网络](http://blog.csdn.net/v_july_v/article/details/51812459)
	* [《DeepLearning》9](http://www.deeplearningbook.org/contents/convnets.html)
	* Pooling

* Supervised learning
	* Structured Data && Unstructured Data 
	* 《DeepLearning》5.7
	* Support Vector Machine
		* [Support Vector Machine (Wiki)](https://en.wikipedia.org/wiki/Support_vector_machine) 
		* 《机器学习》第6章
		* Decision Boundary
		* Kernel

	* Desision tree

* Unsupervised learning
	* 《DeepLearning》5.8 
	* Clustering
		* 《机器学习》第9章
		* K-means
			* 《DeepLearning》5.8.2

	* Dimensionality Reduction
		* 《机器学习》第10章 
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

* Newton's method(牛顿法)

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

注：

《机器学习》代指周志华所著[《机器学习》](https://book.douban.com/subject/26708119/)一书。

《DeepLearning》代指Ian Goodfellow and Yoshua Bengio and Aaron Courville所著[《DeepLearning》](http://www.deeplearningbook.org/)一书。

## Framework

* TensorFlow
	* [First exploration of TensorFlow](http://pengcheng.tech/2017/10/03/first-exploration-of-tensorflow/) 
	* [TensorFlow学习笔记1：入门](http://www.jeyzhang.com/tensorflow-learning-notes.html)
	* [TensorFlow学习笔记2：构建CNN模型](http://www.jeyzhang.com/tensorflow-learning-notes-2.html)

## Academic Papers



## Userful Links:

1. [Machine Learning by Andrew Ng](https://www.coursera.org/learn/machine-learning/home/welcome)

2. [机器学习（周志华）](https://book.douban.com/subject/26708119/)

3. [TensorFlow实战](https://book.douban.com/subject/26974266/)

4. [MNIST数据集](http://yann.lecun.com/exdb/mnist/)

5. [TensorFlow 官方文档中文版](http://wiki.jikexueyuan.com/project/tensorflow-zh/)

6. [Deep Learning（深度学习）学习笔记整理系列](http://blog.csdn.net/zouxy09/article/details/8775360)

7. [台湾大学 李宏毅教案](http://speech.ee.ntu.edu.tw/~tlkagk/courses.html) 

8. [机器学习基石 台湾大学 林轩田](https://www.coursera.org/learn/ntumlone-mathematicalfoundations/home/welcome)

9. [Deep Learning Book](http://www.deeplearningbook.org/)

10. [Deepearning.ai](http://deeplearning.ai/)