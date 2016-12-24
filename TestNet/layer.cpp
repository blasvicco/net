//
//  layer.cpp
//
//  Created by Blas Eugenio Vicco on 11/9/16.
//  Copyright © 2016 Blas Eugenio Vicco. All rights reserved.
//

#include "layer.h"

classLayer::classLayer(unsigned int nip, unsigned int nperce, unsigned int type, double initialMu) {
    input.clear();
    output.clear();
    np = nperce;
    ni = nip;
    mu = initialMu > 0 ? initialMu : mu;
    Perceptron.clear();
    for (unsigned int i = 0; i < np; i++) {
        classPerceptron *tmp = new classPerceptron(nip, type, initialMu);
        Perceptron.push_back(*tmp);
    }
    for (unsigned int i = 0; i < ni; i++) {
        vbias.push_back(Random::dur01() * (double)Random::sign());
    }
}

classLayer::~classLayer() {
    //liberar memoria
}

void classLayer::setInput(vector<double> inp) {
    input.clear();
    input = inp;
}

vector<double> classLayer::getOutput() {
    output = feedForward(input);
    return output;
}

void classLayer::setError(vector<double> error) {
    for (unsigned int i = 0; i < np; i++) {
        Perceptron[i].setError(error[i]);
    }
}

vector<double> classLayer::backFix() {
    vector<double> deltaBackErrors;
    vector<double> deltaBackErrorsTotal(ni, 0.0);
    for (unsigned int i = 0; i < np; i++) {
        deltaBackErrors = Perceptron[i].backFix(input);
        for (unsigned int j = 0; j < ni; j++) {
            deltaBackErrorsTotal[j] += deltaBackErrors[j];
        }
    }
    return deltaBackErrorsTotal;
}

void classLayer::temper(unsigned int k) {
    vector<double> pOMean(output);
    vector<double> pOSample = binomial(pOMean);
    //first gibbs
    vector<double> nIMean = feedBackware(pOSample);
    vector<double> nISample = binomial(nIMean);
    vector<double> nOMean = feedForward(nISample);
    vector<double> nOSample = binomial(nOMean);
    for (unsigned int i = 1; i < k; i++) {
        //gibbs
        vector<double> nIMean = feedBackware(nOSample);
        vector<double> nISample = binomial(nIMean);
        vector<double> nOMean = feedForward(nISample);
        vector<double> nOSample = binomial(nOMean);
    }
    
    for (unsigned int i = 0; i < np; i++) {
        for (unsigned int j = 0; j < ni; j++) {
            double lError = pOMean[i] * input[j] - nOMean[i] * nISample[j];
            Perceptron[i].forFix((lError) / ni, j);
        }
        Perceptron[i].forBiasFix((pOSample[i] - nOMean[i]) / ni);
    }
    
    for (unsigned int i = 0; i < ni; i++) {
        vbias[i] += mu * (input[i] - nISample[i]) / ni;
    }
}

//Private
vector<double> classLayer::feedForward(vector<double> input) {
    vector<double> output;
    double tmp = 0;
    for (unsigned int i = 0; i < np; i++) {
        Perceptron[i].setInput(input);
        tmp = Perceptron[i].getOutput();
        output.push_back(tmp);
    }
    return output;
}

vector<double> classLayer::feedBackware(vector<double> input) {
    vector<double> tmp;
    for (unsigned int i = 0; i< ni; i++) {
        double sum = 0.0;
        for (unsigned int j = 0; j< np; j++) {
            sum += Perceptron[j].feedBackware(input[i], i);
        }
        //should we change this if it is a TANH activation function???
        tmp.push_back(Activation::f(SIGMOID, sum));
    }
    return tmp;
}

vector<double> classLayer::binomial(vector<double> input) {
    vector<double> tmp;
    for (unsigned int i = 0; i < input.size(); i++) {
        //should we change this if it is a TANH activation function???
        tmp.push_back((Random::dur01() > input[i]) ? 1.0 : 0.0);
    }
    return tmp;
}
