### Forward Forward algorithm

Paper: [Geoffrey Hinton. The Forward-Forward Algorithm: Some Preliminary Investigations](https://www.cs.toronto.edu/~hinton/FFA13.pdf)

### 0. This repo.
1. Implemented the forward forward algorithm with the unsupervised examples (paper section 3.2) and the supervised examples (section 3.3) based on my understanding of the paper.

2. Used MNIST data (60000 training samples + 10000 test samples). Hyperparameter searching isn't implemented in this repo.

3. In backprop-based algorithm, we do a forward pass through all layers during which it remembers many intermediate computed results that will be used in the backprop pass to update the layers' weights. In forward-forward algorithm, we do two forward passes on all **hidden layers**, one with positive data and one with negative data. After positive data passes through one layer, a gradient descent is then performed on that layer to optimize for an objective for that particular layer. Then the negative data passes through that layer and performs also a gradient descent on it. These two steps repeat for both data to pass through the next layer, until the last hidden layer. In the output softmax layer, only the positive data is passed through to minimize the cross entropy.

4. The positive and negative data used in each examples will be explained in their sections followed.

5. The objective that each layer is optimzied for is that the goodness for the positive samples is to be close to 1, and that for the negative samples to 0. A goodness function suggested in the paper is the sum of squared activity values minus threshold. The same goodness function is implemented

6. In my implementation, each trainable layer has its own loss function and optimizer.

![goodness function](./images/goodness_function.png)

### 1. Unsupervised example (paper section 3.2)
1. Data: I divided the MNIST dataset into 2 sets. Numbers 0 to 4 are used as positive samples, and the rest as negative samples. This is not how the paper has presented, instead, they augmented the images to generate new images as negative samples.

2. Model: Some hidden layers (Dense or Conv2D) + a softmax layer. The hidden layers are trained with both the positive and the negative samples, whereas the softmax layer uses only the positive samples.

3. The paper says it "use(s) the normalised activity vectors of the last three hidden layers as the inputs to a softmax that is trained to predict the label." This is not how I did it. I simply used the last hidden layer as the inputs to a softmax. However, the result isn't bad.

#### My comparisons 1 (backprop (BP) vs. forward-forward (FF))
Architecture: Dense(16, relu) + Dense(10, relu) + Softmax(10) (Note 1)

- BP accuracy 98.4%,  10 epochs, Adam(lr=0.001)
- FF accuracy 95.1%, 200 epochs, Adam(lr=0.00001) for the Denses, and Adam(lr=0.001) for softmax

#### My comparisons 2
Architecture: Conv2D(4, (8, 8), relu) + Conv2D(8, (6, 6), relu) + Softmax(10) (Note 1)

- BP accuracy 98.6%,  10 epochs, Adam(lr=0.0001)
- FF accuracy 97.3%, 100 epochs, Adam(lr=0.0000001) for the Denses, and Adam(lr=0.00001) for softmax

### 2. Supervised example (paper section 3.3)
1. Data: Divided the MINIST dataset into 2 equal sets for the positive and the negative samples respectively. Replaced negative samples' labels with randomized numbers which is what the paper suggested.

2. Labels: The labels are one-hot encoded and then get overlayed in the first 10 pixels of the images for training. For prediction of an image, the paper suggested 2 approaches, and I use the one that copies the image 10 times and overlays each copied image a different label, then passes these 10 images through the network and whichever has the highest accumulated activity goodness, its label is the prediction.

3. Model: Some hidden layers. No softmax layer. The paper mentioned both approaches with and without a softmax. I am currently more interested in the one without the softmax. The 

4. The paper says "the hidden activities in all but the first hidden layer are then used as the inputs to a softmax that has been learned during training." This is not how I did it. I simply used the last hidden layer as the inputs to a softmax. The result for using Dense layer as the hidden layers is not bad, but that with Conv2D as hidden layers is very poor.

#### My comparisons 1
Architecture: Dense(16, relu) + Dense(10, relu) + Softmax(10) (Note 1, Note 2)

- BP accuracy 91.9%,  10 epochs, Adam(lr=0.001)
- FF accuracy 83.6%, 200 epochs, Adam(lr=0.0001)

#### My comparisons 2
Architecture: Conv2D(4, (8, 8), relu) + Conv2D(8, (6, 6), relu) + Softmax(10) (Note 1, Note 2)

- BP accuracy 92.2%,  10 epochs, Adam(lr=0.0001)
- FF accuracy <12% (no prediction capability)

### 2. Using FF to model top-down effects in perception (paper section 3.4)
To be implemented.

### Notes:
1. Only trainable layers included, go to the script for the full version

2. Softmax is only used in the BP model. FF model does not require a softmax. 







