//
//  layer.cpp
//
//  Created by Blas Eugenio Vicco on 11/9/16.
//  Copyright © 2016 Blas Eugenio Vicco. All rights reserved.
//

#include "layer.h"

classLayer::classLayer(int nip, int nperce, float initialMu) {
    inputs.clear();
    outputs.clear();
    np = nperce;
    ni = nip;
    Perceptron.clear();
    for (int i = 0; i < np; i++) {
        classPerceptron *tmp = new classPerceptron(nip, initialMu);
        Perceptron.push_back(*tmp);
    }
}

classLayer::~classLayer() {
    //liberar memoria
}

void classLayer::setInputs(vector<float> input) {
    inputs.clear();
    inputs = input;
}

vector<float> classLayer::getOutput() {
    outputs.clear();
    float tmp = 0;
    for (int i = 0; i < np; i++) {
        Perceptron[i].setInputs(inputs);
        tmp = Perceptron[i].getOutput();
        outputs.push_back(tmp);
    }
    return outputs;
}

void classLayer::setError(vector<float> error) {
    for (int i = 0; i < np; i++) {
        Perceptron[i].setError(error[i]);
    }
}

vector<float> classLayer::fix() {
    vector<float> deltaBackErrors;
    vector<float> deltaBackErrorsTotal(ni, 0.0);
    for (int i = 0; i < np; i++) {
        deltaBackErrors = Perceptron[i].fix(inputs);
        for (int j = 0; j < ni; j++) {
            deltaBackErrorsTotal[j] += deltaBackErrors[j];
        }
    }
    return deltaBackErrorsTotal;
}
