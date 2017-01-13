//
//  perceptron.cpp
//
//  Created by Blas Eugenio Vicco on 11/9/16.
//  Copyright © 2016 Blas Eugenio Vicco. All rights reserved.
//

#include "perceptron.h"

classPerceptron::classPerceptron(unsigned int nip, unsigned int type, double initialMu, double initialMomentum) {
    activationType = type;
    mu = initialMu > 0 ? initialMu : mu;
    momentum = initialMomentum > 0 ? initialMomentum : 0.0;
    ponderation.clear();
    input.clear();
    output = 0;
    ni = nip;
    double tmp;
    for (unsigned int i = 0; i < ni; i++) {
        tmp = 0;
        tmp = ((Random::dur01()/2.0) + 0.5) * (double)Random::sign();
        ponderation.push_back(tmp);
        lastUpdate.push_back(0.0);
        lastForUpdate.push_back(0.0);
    }
    ponderation.push_back((double)Random::sign()); //adding bias
    lastUpdate.push_back(0.0);
    lastForUpdate.push_back(0.0);
}

classPerceptron::~classPerceptron() {
    //liberar memoria
}

vector<double> classPerceptron::getPonderations() {
    return ponderation;
}

void classPerceptron::setPonderations(vector<double> ponderations) {
    ponderation = ponderations;
}

void classPerceptron::setInput(vector<double> inp) {
    input.clear();
    input = inp;
}

void classPerceptron::setLearningRate(double rate) {
    mu = rate;
}

double classPerceptron::getOutput() {
    double sum = 0;
    for (unsigned int i = 0; i < ni; i++) {
        sum += input[i] * ponderation[i];
    }
    
    sum += ponderation[ni]; //+ bias
    output = Activation::f(activationType, sum);
    return output;
}

void classPerceptron::setError(double error) {
    dEtotaldOutput = Activation::fprime(activationType, output) * error;
}

vector<double> classPerceptron::backFix(vector<double> inp) {
    double delta = dEtotaldOutput;
    vector<double> deltaBackErrors;
    for (unsigned int i = 0; i < ni; i++) {
        double tmp = mu * delta * inp[i] + (momentum * lastUpdate[i]);
        if ((ponderation[i] + tmp > -DBL_MAX) && (ponderation[i] + tmp < DBL_MAX)) {
            ponderation[i] += tmp;
            lastUpdate[i] = tmp;
        }
        deltaBackErrors.push_back(delta * ponderation[i]);
    }
    ponderation[ni] += mu * delta;
    return deltaBackErrors;
}

double classPerceptron::feedBackware(double output, unsigned int wIndex) {
    return ponderation[wIndex] * output;
}

void classPerceptron::forFix(double error, unsigned int wIndex) {
    double tmp = mu * error + (momentum * lastForUpdate[wIndex]);
    if ((ponderation[wIndex] + tmp > -DBL_MAX) && (ponderation[wIndex] + tmp < DBL_MAX)) {
        ponderation[wIndex] += tmp;
        lastForUpdate[wIndex] = tmp;
        
    }
}

void classPerceptron::forBiasFix(double error) {
    forFix(error, ni);
}


