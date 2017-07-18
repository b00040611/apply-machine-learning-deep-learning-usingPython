# Apply Machine Learning and Deep Learning Tools in Our Research (using Python)

A curated list of deep learning resources for computer vision, inspired by [awesome-php](https://github.com/ziadoz/awesome-php) and [awesome-computer-vision](https://github.com/jbhuang0604/awesome-computer-vision).

Maintainers - [Luke Liu](https://github.com/b00040611)

## Contributing

## Table of Contents
- [Python](#python)
  - [Python IDE](#python-ide)
  - [Courses](#courses)
  - [Python Books](#python-books)
  - [Other Resources](#other-resources)
- [Machine Learning Using Python](#machine-learning-using-python)
  - [Machine Learning Courses](#machine-learning-courses)
  - [Python Packages for Machine Learning](#python-packages-for-machine-learning)
  - [Practice Your Skills](#pratice-your-skills)
- [Deep Learning](#deep-learning)
  - [Deep Learning Courses](#deep-learning-courses)
  - [Frameworks](#frameworks)
  - [Network Architectures and Examples](#network-architectures-and-examples)
    - [Convolutional Networks](#convolutional-networks)
    - [Recurrent Neural Networks](#recurrent-neural-networks)
    - [Recursive Neural Networks](#recursive-neural-networks)
  - [Datasets](#datasets)
    -[Image](#image)
    -[Handwriting](#handwriting)
    -[Video](#video)
  - [Model Zoo](#model-zoo)
- [Using Commercial Cloud Computing Platforms](#using-commercial-cloud-computing-platforms)
  - [ML and DL APIs](#ml-and-dl-apis)

## Python

### Python IDE

* [Anaconda](https://www.continuum.io/downloads)- The open source version of Anaconda is a high performance distribution of Python and R and includes over 100 of the most popular Python, R and Scala packages for data science.

### Courses

* [Programming for Everybody](https://www.coursera.org/learn/python) - Open course in Coursera from University of Michigan.

### Python Books

* [Python Cookbook](https://learko.github.io/books/coding/Python_Cookbook_3rd_Edition.pdf) - Advanced use of python.

### Other Resources

* [A list of awesome Python frameworks, libraries and software](https://github.com/uhub/awesome-python)
* [Another list of awesome Python frameworks, libraries and software](https://github.com/vinta/awesome-python#distribution)
* [List of Python API Wrappers](https://github.com/realpython/list-of-python-api-wrappers)
* [List of resources about Python in Education](https://github.com/quobit/awesome-python-in-education)
* [Some frameworks including Django and Flask](https://github.com/adrianmoisey/learn-python)

## Machine Learning Using Python

### Machine Learning Courses

* [Machine Learning](https://www.coursera.org/learn/machine-learning) - Introduction to machine learning, datamining, and statistical pattern recognition, by Andrew Ng (Standford University).
* [Tutorials: Data analysis using Python](https://pythonprogramming.net/data-analysis-tutorials/) - Tutorial with code samples.

### Python Packages for Machine Learning

* [scikit-learn](http://scikit-learn.org/stable/) - Simple and efficient tools for data mining and data analysis.
* [pandas](http://pandas.pydata.org/) - Python Data Analysis Library.
* [Numpy](http://www.numpy.org/) - Fundamental package for scientific computing with Python.
* [SciPy](https://www.scipy.org/) - user-friendly and efficient numerical routines such as routines for numerical integration and optimization.
* [matplotlib](https://matplotlib.org/) - Matplotlib is a Python 2D plotting library which produces publication quality figures in a variety of hardcopy formats and interactive environments across platforms.
* [list of other python libraries for machine learning](https://github.com/josephmisiti/awesome-machine-learning#python)

### Practice Your Skills

* [Pratice your skills and learn from others on kaggle](https://www.kaggle.com/)

## Deep Learning
### Deep Learning Courses
  * [University of Toronto] [Neural Networks for Machine Learning](https://www.coursera.org/learn/neural-networks) - by:  Geoffrey Hinton, Professor
  * [Stanford] [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/) - by Fei-Fei Li, Andrej Karpathy
  * [Stanford] [CS224d: Deep Learning for Natural Language Processing](http://cs224d.stanford.edu/)

### Frameworks
* Keras: The Python Deep Learning library on top of either TensorFlow, CNTK or Theano. [[Web](https://keras.io/)]
* Tensorflow (Google): An open source software library for numerical computation using data flow graph by Google [[Web](https://www.tensorflow.org/)]
* Torch7 (NYU/Facebook): Deep learning library in Lua, used by Facebook and Google Deepmind [[Web](http://torch.ch/)]
  * Torch-based deep learning libraries: [[torchnet](https://github.com/torchnet/torchnet)],
  * Pytorch (Facebook): Tensors and Dynamic neural networks in Python with strong GPU acceleration.[[Web](http://pytorch.org/)]
* Caffe (UC Berkeley): Deep learning framework by the BVLC [[Web](http://caffe.berkeleyvision.org/)]
  *Caffe2 (Facebook): A New Lightweight, Modular, and Scalable Deep Learning Framework[[Web](https://caffe2.ai/)]
* Theano (U Montreal): Mathematical library in Python, maintained by LISA lab [[Web](http://deeplearning.net/software/theano/)]
  * Theano-based deep learning libraries: [[Pylearn2](http://deeplearning.net/software/pylearn2/)], [[Blocks](https://github.com/mila-udem/blocks)], [[Keras](http://keras.io/)], [[Lasagne](https://github.com/Lasagne/Lasagne)]
* CNTK (Microsoft): A unified deep-learning toolkit by Microsoft [[Web](https://docs.microsoft.com/en-us/cognitive-toolkit/)]
* MXNet (Amazon): Developed by U Washington, CMU, MIT, Hong Kong U, etc but main framework of choice at AWS [[Web](https://github.com/dmlc/mxnet/)]

### Network Architectures and Examples

#### Convolutional Networks

(CNNs are a specialized kind of neural network for processing datathat has a known, grid-like topology. Examples include time-series data, which canbe thought of as a 1D grid taking samples at regular time intervals, and image data,which can be thought of as a 2D grid of pixels. The name “convolutional neuralnetwork” indicates that the network employs a mathematical operation called convolution. Convolution is a specialized kind of linear operation. Convolutionalnetworks are simply neural networks that use convolution in place of general matrix multiplication in at least one of their layers. Typical applications: Image Classification, Image Segmentation\Object Detection [Region-CNN])

* AlexNet [[Paper]](http://papers.nips.cc/book/advances-in-neural-information-processing-systems-25-2012)
  * Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton, ImageNet Classification with Deep Convolutional Neural Networks, NIPS, 2012.
* VGG-Net [[Web]](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) [[Paper]](http://arxiv.org/pdf/1409.1556)
  * Karen Simonyan and Andrew Zisserman, Very Deep Convolutional Networks for Large-Scale Visual Recognition, ICLR, 2015.
* GoogLeNet [[Paper]](http://arxiv.org/pdf/1409.4842)
  * Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich, CVPR, 2015.
* Microsoft ResNet (Deep Residual Learning) [[Paper](http://arxiv.org/pdf/1512.03385v1.pdf)][[Slide](http://image-net.org/challenges/talks/ilsvrc2015_deep_residual_learning_kaiminghe.pdf)]
  * Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, Deep Residual Learning for Image Recognition, arXiv:1512.03385.
* R-CNN, UC Berkeley [[Paper-CVPR14]](http://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.pdf) [[Paper-arXiv14]](http://arxiv.org/pdf/1311.2524)
* Fast R-CNN, Microsoft Research [[Paper]](http://arxiv.org/pdf/1504.08083)
  * Ross Girshick, Fast R-CNN, arXiv:1504.08083.
* Faster R-CNN, Microsoft Research [[Paper]](http://arxiv.org/pdf/1506.01497)
  * Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun, Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks, arXiv:1506.01497.
* Inside-Outside Net [[Paper]](http://arxiv.org/abs/1512.04143)
  * Sean Bell, C. Lawrence Zitnick, Kavita Bala, Ross Girshick, Inside-Outside Net: Detecting Objects in Context with Skip Pooling and Recurrent Neural Networks
* Deep Residual Network (Current State-of-the-Art) [[Paper]](http://arxiv.org/abs/1512.03385)
  * Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, Deep Residual Learning for Image Recognition
* R-FCN [[Paper]](https://arxiv.org/abs/1605.06409) [[Code]](https://github.com/daijifeng001/R-FCN)
  * Jifeng Dai, Yi Li, Kaiming He, Jian Sun, R-FCN: Object Detection via Region-based Fully Convolutional Networks

#### Recurrent Neural Networks

(RNNs are a family ofneural networks for processing sequential data. Typical applications: Image Captioning[CNN+RNN], Natural Language Processing. Common used techniques are LSTM and GRU which are variants of RNN.)

* LSTM[[Paper](https://arxiv.org/pdf/1503.04069.pdf?utm_content=buffereddc5&utm_medium=social&utm_source=plus.google.com&utm_campaign=buffer)] LSTM: A search space odyssey (2016), K. Greff et al.
* GRU [[Paper](https://arxiv.org/pdf/1406.1078.pdf)] Learning phrase representations using RNN encoder-decoder for statistical machine translation (2014), K. Cho et al.

#### Recursive Neural Networks

(Recursive neural networks2 represent yet another generalization of recurrent networks,
with a different kind of computational graph, which is structured as a deep
tree, rather than the chain-like structure of RNNs.)

* RNN[[Paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.383.1327&rep=rep1&type=pdf)]Recursive deep models for semantic compositionality over a sentiment treebank (2013), R. Socher et al.
### Resources List of Deep Learning

* [Awesome Deep Learning](https://github.com/ChristosChristofidis/awesome-deep-learning) -  Practical Deep Learning resources For Coders
* [Awesome Deep Vision](https://github.com/apacha/awesome-deep-vision) - A list of deep learning resources for computer vision
* [List of Most Cited Deep Learning Papers](https://github.com/terryum/awesome-deep-learning-papers)
* [A list of recent papers and trained models](https://github.com/endymecy/awesome-deeplearning-resources)


### Datasets

#### Image
* [IMAGENET](http://www.image-net.org/)
* [CIFAR-10 and CIFAR-100](http://www.cs.toronto.edu/~kriz/cifar.html)
* [AT&T Laboratories Cambridge face database](http://www.uk.research.att.com/facedatabase.html)
* [Google House Numbers](http://ufldl.stanford.edu/housenumbers/) from street view
* [Flickr Data](https://yahooresearch.tumblr.com/post/89783581601/one-hundred-million-creative-commons-flickr-images) 100 Million Yahoo dataset

#### Handwriting
* [MNIST](http://yann.lecun.com/exdb/mnist/) Handwritten digits

#### Video
* [YouTube-8M Dataset](https://research.google.com/youtube8m/) - YouTube-8M is a large-scale labeled video dataset that consists of 8 million YouTube video IDs and associated labels from a diverse vocabulary of 4800 visual entities.

### Model Zoo

* [Caffe Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
* [MXNet Model Zoo](http://mxnet.io/model_zoo/)
* [Keras Model Zoo](https://keras.io/applications/)
* [Torch Model Zoo](https://github.com/szagoruyko/loadcaffe)- Maintains a list of popular models like AlexNet and VGG .Weights ported from Caffe

## Using Commercial Cloud Computing Platforms