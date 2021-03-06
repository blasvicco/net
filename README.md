# Net
This is an Artificial Neural Network implemented on C++ using Object Oriented Programming.
The library allow us to use just a perceptron or a complete multilayer network.
Two activation functions were developed on this libs, SIGMOID and TANH.
Also a pretraining method based on Restricted Boltzmann Machine (RMB) was added.

### Using the library

Be sure that all the file are included in your project.

In the code where you want to use it just include the Net header file.

```c++
#include "net.h"
#include "activation.h"
```

Then you can instantiate and initialize a network.

```c++
void yourFunction(type someParameter) {
	classNet net = *new classNet(); //instantiation
	vector<int> numberOfPerceptronPerLayer = {4, 6, 8, 4}; //helper to initialize the network
	int ni = 2; //number of inputs
	int no = 2; //number of outputs
	float mu = 0.6; //learning rate
	net.ini(ni, no, numberOfPerceptronPerLayer, TANH, mu);

	vector<float> inputs = {-1, -1}; //the input values
	net.setInputs(inputs); //to set the inputs to the net
	vector<float> output(net.getOutput()); //get the output after the forward propagation
	//for the back propagation we need to define the error
	vector<float> wished = {1, 1}; //expected outputs
	vector<float> errors = {wished[0] - output[0], wished[1] - output[1]}; //errors
	net.backFix(errors); //back propagation - learning!!!
}
```

### XOR example

Check the video: https://youtu.be/psMW2K3iJ_s

The main.cpp file implements a complete well known XOR example.
You will see first in the code the definition of important constants.

```c++
const unsigned int TYPE = TANH;//SIGMOID; //TANH
const bool RBMPRETRAINING = true; //swap from false to true to activate the pre training
const double MINERRORACCEPTED = (TYPE == SIGMOID) ? 0.03 : 0.003;
const double LEARNINGRATERBM = 0.8;
const double LEARNINGRATEMLP = 0.3;
const double MOMENTUM = 0.2;
```

TYPE is the activation function that we will use for each perceptron in the Net. You can choose between SIGMOID and TANH.

RBMPRETRAINING is a boolean value that activates or deactivates the pre training using RBM.

MINERRORACCEPTED is a double value used to stop the training process when the Global Error reaches the threshold. Remember that higher values will result in less training iterations but maybe it will not be enough to find a good solution. Smaller values will increment the number of iteration significantly.

LEARNINGRATERBM and LEARNINGRATEMLP are the gradient damping for the learning process and MOMENTUM rate allows the attenuation of oscillations in the gradient descent.

Inside the main function we define the numEpoch. For these experiments we set it on 50 iterations. For each epoch we will:

Initialize a network
Pre train
Train
Test
By default we will print for each epoch, the number of iterations needed by the Net to get trained and the total number of fails in the test phase. At the end of all epochs, we will print the average of iteration and the average of porcentual errors to have a better approach to the Network configuration response.

For the XOR problem we will have two inputs and one output. These values will be different depending of the activation function we chose for the Net. For SIGMOID the inputs and the output will be 0 or 1 and for TANH the range will be -1, 1.

The XOR problem doesn't need more than one hidden layer to be solved. You can test it deactivating the pre training and setting the network as:

```c++
const bool RBMPRETRAINING = false;
const double LEARNINGRATEMLP = 0.3;
vector<unsigned int> nppl = {4};
//NETWORK SETTING
//INPUT, OUTPUT, NUMBER PERCEPTRON PER LAYER, ACTIVATION FUNCTION, LEARNING RATE
net.ini(2, 1, nppl, TYPE, LEARNINGRATERBM, MOMENTUM);
net.setLearningRate(LEARNINGRATEMLP);
```

Response example:
```
 ----------------------------------------
 Min error reached at iteration avg: 19409
 Average error: 0.000% --> Perfect Response
 ----------------------------------------
```

But in order to test the RBM we will add extra hidden layers to the Net. You can see that if we add these extra layers and we do not activate the pre training, the response from the Net to solve the XOR problem is not good.

```c++
const bool RBMPRETRAINING = false;
const double LEARNINGRATEMLP = 0.3;
vector<unsigned int> nppl = {6, 4, 2};
//NETWORK SETTING
//INPUT, OUTPUT, NUMBER PERCEPTRON PER LAYER, ACTIVATION FUNCTION, LEARNING RATE
net.ini(2, 1, nppl, TYPE, LEARNINGRATERBM, MOMENTUM);
net.setLearningRate(LEARNINGRATEMLP);
```

Response example adding more than one hidden layer and without pre training:
```
 ----------------------------------------
 Min error reached at iteration avg: 13660
 Average error: 10.098%
 ----------------------------------------
```

Incrementing the learning rate doesn't help:

```c++
const bool RBMPRETRAINING = false;
const double LEARNINGRATEMLP = 0.8;
vector<unsigned int> nppl = {6, 4, 2};
//NETWORK SETTING
//INPUT, OUTPUT, NUMBER PERCEPTRON PER LAYER, ACTIVATION FUNCTION, LEARNING RATE
net.ini(2, 1, nppl, TYPE, LEARNINGRATERBM, MOMENTUM);
net.setLearningRate(LEARNINGRATEMLP);
```

Response example without pre training and incrementing the learning rate:
```
 ----------------------------------------
 Min error reached at iteration avg: 12332
 Average error: 44.976%
 ----------------------------------------
```

If we activate the RBM:
```c++
const bool RBMPRETRAINING = true;
const double LEARNINGRATERBM = 0.8;
const double LEARNINGRATEMLP = 0.8;
vector<unsigned int> nppl = {6, 4, 2};
//NETWORK SETTING
//INPUT, OUTPUT, NUMBER PERCEPTRON PER LAYER, ACTIVATION FUNCTION, LEARNING RATE
net.ini(2, 1, nppl, TYPE, LEARNINGRATERBM, MOMENTUM);
net.setLearningRate(LEARNINGRATEMLP);
```

Response example with pre training:
```
 ----------------------------------------
 Min error reached at iteration avg: 8325
 Average error: 4.874%
 ----------------------------------------
 ```
 
And if we refine the tuning decreasing the learning rate for the MLP again to 0.3:
 ```c++
const bool RBMPRETRAINING = true;
const double LEARNINGRATERBM = 0.8;
const double LEARNINGRATEMLP = 0.3;
vector<unsigned int> nppl = {6, 4, 2};
//NETWORK SETTING
//INPUT, OUTPUT, NUMBER PERCEPTRON PER LAYER, ACTIVATION FUNCTION, LEARNING RATE
net.ini(2, 1, nppl, TYPE, LEARNINGRATERBM, MOMENTUM);
net.setLearningRate(LEARNINGRATEMLP);
```

Response example:
```
 ----------------------------------------
 Min error reached at iteration avg: 17517
 Average error: 0.958%
 ----------------------------------------
 ```
 
The last response is even better than the same configuration without pre training.
We saw in these experiments the problem of back propagating the error between hidden layers. The more hidden layers we add, the greater the vanishing gradient problem we have. So, in order to sort this issue we add the pre training with the RBM.


As you can see in the code, the pre training is a loop of 1000 samples. Also, we are applying the RBM just to the first hidden layer and we are setting just one Gibbs steps. Please feel free to play with this parameters.

```c++
//PRE TRAINING WITH RBM -> Restricted Boltzmann Machine
if (RBMPRETRAINING) {
	//pre train with 1000 samples
	for (unsigned int i = 0; i < 1000; i++) {
		inputs = {s(), s()};
		wish = {xxor(inputs)};
		net.setInput(inputs);
		//temper(deep, k)
		//deep inside net (1 == first layer)
		//k = gibbs iteration -> I don't see any improve with k == 100 or k == 1
		net.temper(1, 1);
	}
}
```

After the pre training we perform the well know MLP training or back-propagation. The loop here has two conditions, a hardcoded limit in 20000 iterations or the gError (global error) being minor than the threshold. The hardcoded limit is a safe infinite loop breaker for Net configurations that doesn't converge or doesn't reach the threshold. The (gError > MINERRORACCEPTED) condition is not the best to end the training loop. It could happen that the MINERRORACCEPTED is reached with no enough training iterations, resulting in a bad Net configuration. Please, feel free to implement a better approach.

In order to test the Net, I decided to calculate the error as wish - round(output). Where round(output) will give us the logic response on/off. I also set the test loop to 1000 but you can change it easily defined the var ctest.

```c++
unsigned int cwrong = 0;
unsigned int ctest = 1000;
for (unsigned int n = 0; n < ctest; n++) {
	inputs = {s(), s()};
	wish = {xxor(inputs)};
	net.setInput(inputs);
	vector<double> output = net.getOutput();
	vector<double> error = {wish[0] - round(output[0])};
	gError = pow(pow(error[0], 2), 0.5);
	cwrong += (gError != 0) ? 1 : 0;
}
```
