# Net
This is an Artificial Neural Network implemented on c++ using Object Oriented Design pattern.
The library allow us to use just a perceptron or a complete multilayer network.
The activation function for the perceptrons is the Sigmoid function by default (feel free to implement yours).

##### Some of the technologies used in this project are:
  - C++
  - Math.h


### Using the library

Be sure that all the file are included in your project.

In the code where you want to use it you need to include the Net header file.

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
	float mu = 0.6; //learning factor
	net.ini(ni, no, numberOfPerceptronPerLayer, TANH,mu);

	vector<float> inputs = {-1, -1}; //the input values
	net.setInputs(inputs); //to set the inputs to the net
	vector<float> output(net.getOutput()); //get the output after the forward propagation
	//for the back propagation we need to define the error
	vector<float> wished = {1, 1}; //expected outputs
	vector<float> errors = {wished[0] - output[0], wished[1] - output[1]}; //errors
	net.fix(errors); //back propagation - learning!!!
}
```

Please refer to the main.cpp file to see an example of how to use the ANN.
