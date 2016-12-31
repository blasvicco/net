# Net
This is an Artificial Neural Network implemented on C++ using Object Oriented Programming.
The library allow us to use just a perceptron or a complete multilayer network.
Two activation functions were developed on this libs, SIGMOID and TANH.
Also a pretraining method based on Restricted Boltzmann Machine (RMB) was added.

### Using the library

Be sure that all the file are included in your project.

In the code where you want to use it just include the Net header file.

```c++
#include “net.h”
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

The main.cpp file implements a complete well known XOR example.
You will see first on the code the definition of 4 important constants.

```c++
const unsigned int TYPE = TANH;//SIGMOID; //TANH
const bool RBMPRETRAINING = true; //swap from false to true to activate the pre training
const double MINERRORACCEPTED = (TYPE == SIGMOID) ? 0.03 : 0.005;
const double LEARNINGRATE = 0.8;
```

TYPE is the activation function that we will use for the perceptron on the Net. You can chose between SIGMOID and TANH.
RBMPRETRAINING is a boolean value that activate or deactivate the pretraining using RBM.
MINERRORACCEPTED is a double value used to stop the training process when the Global Error reach the threshold. Remember that higher values will result in less training iterations but maybe It will no be enough to find a good solution. Smaller values will incress the number of iteration significantly.
LEARNINGRATE is the gradient damping for the learning process. At the moment, the value apply for both, the pretraining and the backpropagation.

Inside the main function we define the numEpoch, for this example we set it on 50 iterations.
For each epoch we will:
* Initialize a network
* Pre training
* Training
* Test

By default we will print in each epoch, the number of iteration needed for the Net to get trained and the total of fails in the test phase.
At the end of the epoches iteration we will print the average of iteration and the average of percentual errors.
In this way we will have a better approach of the Network configuration response.

For the XOR problem we will have two inputs and one output. These values will be different depending of the activation function we chose for the Net.
For SIGMOID inputs and output will be 0 or 1 and for TANH the range will be -1, 1.

The XOR problem doesn't need more than one hidden layer to be solved. You can test it deactivating the pre training and setting the network as:

```c++
const bool RBMPRETRAINING = false;
const double LEARNINGRATE = 0.3;
vector<unsigned int> nppl = {3};
//NETWORK SETTING
//INPUT, OUTPUT, NUMBER PERPECTRON PER LAYER, ACTIVATION FUNCTION, LEARNING RATE
net.ini(2, 1, nppl, TYPE, LEARNINGRATE);
```
Response example:
```
----------------------------------------
Min error reached at iteration avg: 13633
Average error: 1.566%
----------------------------------------
```

But in order to test the RBM we will add an extra hidden layer to the Net. You can see that if we add this extra layer, incress the learning rate to 0.8 and we do not activate the pre training, the response of the Net to solve the XOR problem is not good at all.

```c++
const bool RBMPRETRAINING = false;
const double LEARNINGRATE = 0.8;
vector<unsigned int> nppl = {4, 3};
//NETWORK SETTING
//INPUT, OUTPUT, NUMBER PERPECTRON PER LAYER, ACTIVATION FUNCTION, LEARNING RATE
net.ini(2, 1, nppl, TYPE, LEARNINGRATE);
```

Response example without pre training:
```
----------------------------------------
Min error reached at iteration avg: 7689
Average error: 29.176%
----------------------------------------
```

Versus:

```c++
const bool RBMPRETRAINING = true;
const double LEARNINGRATE = 0.8;
vector<unsigned int> nppl = {4, 3};
//NETWORK SETTING
//INPUT, OUTPUT, NUMBER PERPECTRON PER LAYER, ACTIVATION FUNCTION, LEARNING RATE
net.ini(2, 1, nppl, TYPE, LEARNINGRATE);
```

Response example with pre training:
```
----------------------------------------
Min error reached at iteration avg: 5494
Average error: 10.988%
----------------------------------------
```

As you can see in the code, the pre training is a loop of 1000 samples. Also, we are applying the RBM just to the first hidden layer and we are setting just one Gibbs steps. Please feel free to play with this parameters.

```c++
//PRE TRAINING WITH RBM -> Restricted Boltzman Machine
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

After the pre training we performe the well know MLP training or backpropagation. The loop here has two conditions, a hardcoded limit in 20000 iterations or the gError (global error) being minor than the threshold.
The hardcoded limit is a safe infinite loop breaker for Net configurations that doesn't converge or doesn't reach the threshold.
The (gError > MINERRORACCEPTED) condition is not the best to end the training loop. It could happen that the MINERRORACCEPTED is reached with no enough training iterations, resulting in a bad Net configuration. Please, feel free to implement a better approach.

In order to test the Net I decided to calculate the error as wish - round(output). Where round(output) will give us the logic response on/off. I also set the test loop to 1000 but you can change it easily defined the var ctest.

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
