Download Link: https://assignmentchef.com/product/solved-cv1-lab-final-part-2-cnns-for-image-classification
<br>
This part of the assignment makes use of Convolutional Neural Networks (CNN). The previous part makes use of hand-crafted features like SIFT to represent images, then trains a classifier on top of them. In this way, learning is a two-step procedure with image representation and learning. The method used here instead <em>learns </em>the features jointly with the classification. Training CNNs roughly consists of three parts: (i) Creating the network architecture, (ii) Reprocessing the data, (iii) Feeding the data to the network, and updating the parameters. Please follow the instruction and finish the below tasks. (<strong>Note: </strong>You do not need to strictly follow the structure/functions of the provided script.)

<h1>1           Session 1: Image Classification on CIFAR-10</h1>

<h2>1.1         Installation</h2>

First of all, you need to install PyTorch and relevant packages. In this session, we will use CIFAR-10 as the training and testing dataset.

<h3>CIFAR-10 (3-<em>pts</em>)</h3>

The relevant script is provided in <em>Lab project part2.pynb</em>. You need to run and modify the given code and <strong>show </strong>the example images of CIFAR-10, <strong>describe </strong>the classes and images of CIFAR-10. (Please visualize at least one picture for each class.)

<h2>1.2         Architecture understanding</h2>

In this section, we provide two wrapped classes of architectures defined by <em>nn.Module</em>. One is an ordinary two-layer network (<em>TwolayerNet</em>) with fully connected layers and ReLu, and the other is a Convolutional Network (<em>ConvNet</em>) utilizing the structure of LeNet-5[2].

<h3>Architectures (5-<em>pts</em>)</h3>

<ol>

 <li>Complement the architecture of <em>TwolayerNet </em>class, and complement the architecture of <em>ConvNet </em>class using the structure of LeNet-5[2]. (3-<em>pts</em>)</li>

 <li>Since you need to feed color images into these two networks, what’s the kernel size of the first convolutional layer in <em>ConvNet</em>? and how many trainable parameters are there in ”F6” layer (given the calculation process)? (2-<em>pts</em>)</li>

</ol>

<h2>1.3         Preparation of training</h2>

In above section, we use the <em>CIFAR10 </em>dataset class from <em>torchvision.utils </em>provided by PyTorch. Whereas in most cases, you need to prepare the dataset yourself. One of the ways is to create a <em>dataset </em>class yourself and then use the <em>DataLoader </em>to make it iterable. After preparing the training and testing data, you also need to define the transform function for data augmentation and optimizer for parameter updating.

<h2>1.4            Setting up the hyperparameters</h2>

Some parameters must be set properly before the training of CNNs. These parameters shape the training procedure. They determine how many images are to be processed at each step, how much the weights of the network will be updated, how many iterations will the network run until convergence. These parameters are called hyperparameters in the machine learning literature.

<h3>Hyperparameter Optimization and Evaluation (10-<em>pts</em>)</h3>

<ol>

 <li>Play with ConvNet and TwolayerNet yourself, set up the hyperparameters, and reach the accuracy as high as you can. You can modify the <em>train</em>, <em>Dataloader</em>, <em>transform </em>and <em>Optimizer </em>function as you like.</li>

 <li>You can also modify the architectures of these two Nets. Let’s add 2 more layers in ”TwolayerNet” and ConvNet, and show the results. (You can decide the size of these layers and where to add them.) Will you get higher performances? explain why.</li>

 <li>Show the final results and described what you’ve done to improve the results. Describe and explain the influence of hyperparameters among <em>TwolayerNet </em>and <em>ConvNet</em>.</li>

 <li>Compare and explain the differences of these two networks regarding the architecture, performances, and learning rates.</li>

</ol>

<h3>Hint</h3>

You can adjust the following parameters and other parameters not listed as you like: <em>Learning rate, Batch size, Number of epochs, optimizer, transform function, Weight decay etc. </em>You can also change the structure a bit, for instance, adding Batch Normalization[4] layers. Please do not use external well-defined networks and please do not add more than 3 additional (beyond the original network) convolutional layers.

<h1>2              Session 2: Fine-tuning the ConvNet</h1>

In the previous session, the above-implemented network (ConvNet) is trained on a dataset named CIFAR-10, which contains the images of 10 different object categories. The size of each image is 32 × 32 × 3. In this session, we will use a subset of STL-10 with <strong>larger sizes </strong>and <strong>different object classes</strong>. Consequently, there is a discrepancy between the dataset we used to train (CIFAR-10) and the new dataset (STL-10). One of the solutions is to train the whole network from scratch. However, the number of parameters is too large to be trained properly with such few numbers of images provided from STL-10. Another solution is to shift the learned weights in a way to perform well on the test set, while preserving as much information as necessary from the training class. This procedure is called transfer learning and has been widely used in the literature. Fine-tuning is often used in such circumstances, where the weights of the pre-trained network change gradually. One of the ways of fine-tuning is to use the same architectures in all layers except the output layer, as the number of output classes changes (<strong>from 10 to 5</strong>).

<h2>2.1         STL-10 Dataset</h2>

<h2>2.2         Fine-tuning ConvNet</h2>

In this case, you need to modify the output layer of pre-trained ConvNet module from 10 to 5. In this way, you can either load the pre-trained parameters and then modify the output layer or change the output layer firstly and then load the matched pre-trained parameters. You can find the examples from <a href="https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html">https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html</a> and <a href="https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html">https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html</a><a href="https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html">.</a>

<h2>2.3         Bonus (optional)</h2>

<h1>References</h1>

<ul>

 <li>LeCun Y, Bottou L, Bengio Y, et al. Gradient-based learning applied to document recognition[J]. Proceedings of the IEEE, 1998, 86(11): 2278-2324.</li>

 <li>LeCun, Yann, et al. ”Gradient-based learning applied to document recognition.” Proceedings of the IEEE 86.11 (1998): 2278-2324.</li>

 <li>Sharif Razavian, Ali, et al. ”CNN features off-the-shelf: an astounding baseline for recognition.” Proceedings of the IEEE conference on computer vision and pattern recognition workshops. 2014.</li>

 <li>Ioffe, Sergey, and Christian Szegedy. ”Batch normalization: Accelerating deep network training by reducing internal covariate shift.” International conference on machine learning. PMLR, 2015.</li>

</ul>