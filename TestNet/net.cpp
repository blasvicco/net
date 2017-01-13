//
//  net.cpp
//
//  Created by Blas Eugenio Vicco on 11/9/16.
//  Copyright © 2016 Blas Eugenio Vicco. All rights reserved.
//

#include "net.h"

classNet::classNet() {
    
}

classNet::~classNet() {
    //liberar memoria
}

void classNet::ini(unsigned int ninput, unsigned int noutput, vector <unsigned int> numppl, unsigned int type, double initialMu, double initialMomentum) {
    classLayer *tmp;
    input.clear();
    output.clear();
    nppl.clear();
    nl = (unsigned int)numppl.size();
    nppl = numppl;
    ni = ninput;
    no = noutput;
    Layer.clear();
    if (nl == 0) {
        tmp = new classLayer(ninput, noutput, type, initialMu, initialMomentum);
        Layer.push_back(*tmp);
    } else {
        tmp = new classLayer(ninput, nppl[0], type, initialMu, initialMomentum);
        Layer.push_back(*tmp);
        for (unsigned int i = 0; i < nl - 1; i++) {
            tmp = new classLayer(nppl[i], nppl[i + 1], type, initialMu, initialMomentum);
            Layer.push_back(*tmp);
        }
        tmp = new classLayer(nppl[nl - 1], noutput, type, initialMu, initialMomentum);
        Layer.push_back(*tmp);
    }
    nl = (unsigned int)Layer.size();
}

void classNet::clone(classNet Net, double mRate) {
    Layer = Net.getLayer();
    nl = (unsigned int)Layer.size();
    ni = Net.getNi();
    no = Net.getNo();
    mutate(getGens(), mRate);
}

vector <classLayer> classNet::getLayer() {
    return Layer;
}

unsigned int classNet::getNi() {
    return ni;
}

unsigned int classNet::getNo() {
    return no;
}

void classNet::setInput(vector<double> inp) {
    input = inp;
}

void classNet::setLearningRate(double rate) {
    for (unsigned int i = 0; i < nl; i++) {
        Layer[i].setLearningRate(rate);
    }
}

vector<double> classNet::getOutput() {
    output.clear();
    feedForward(nl);
    return output;
}

void classNet::backFix(vector <double> error) {
    int index = nl - 1;
    while(index >= 0) {
        Layer[index].setError(error);
        error = Layer[index].backFix();
        index--;
    }
}

void classNet::temper(unsigned int deep, unsigned int k) {
    for (unsigned int i = 0; i < deep; i++) {
        feedForward(i+1);
        Layer[i].temper(k);
    }
}

//Private
vector<double> classNet::getGens() {
    vector<double> gens = {};
    for (unsigned int i = 0; i < nl; i++) {
        vector<classPerceptron> perceptrons = Layer[i].getPerceptron();
        for (unsigned int j = 0; j < perceptrons.size(); j++) {
            vector<double> tmp = perceptrons[j].getPonderations();
            gens.insert(gens.end(), tmp.begin(), tmp.end());
        }
    }
    return gens;
}

void classNet::setGens(vector<double> gens) {
    unsigned int index = 0;
    for (unsigned int i = 0; i < nl; i++) {
        vector<classPerceptron> perceptrons = Layer[i].getPerceptron();
        for (unsigned int j = 0; j < perceptrons.size(); j++) {
            vector<double> ponderations = perceptrons[j].getPonderations();
            for (unsigned int k = 0; k < ponderations.size(); k++) {
                ponderations[k] = gens[index];
                index++;
            }
            perceptrons[j].setPonderations(ponderations);
        }
    }
}

void classNet::mutate(vector<double> gens, double mRate) {
    vector<unsigned int> tmp;
    unsigned int nGensToMutate = (unsigned int) (gens.size() * mRate);
    unsigned int index = 0;
    while (nGensToMutate > 0) {
        if (mRate != 1.0) {
            index = Random::rndRange(0, (int)gens.size() - 1);
            while (find(tmp.begin(), tmp.end(), index) != tmp.end()) {
                index = Random::rndRange(0, (int)gens.size() - 1);
            }
            tmp.push_back(index);
        }
        gens[index] = ((Random::dur01()/2.0) + 0.5) * (double)Random::sign(); //mutation
        nGensToMutate--;
        index++;
    };
    setGens(gens);
}

void classNet::feedForward(unsigned int deep) {
    output = input;
    for (unsigned int i = 0; i < deep; i++) {
        Layer[i].setInput(output);
        output = Layer[i].getOutput();
    }
}
