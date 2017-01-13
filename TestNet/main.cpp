
//
//  main.cpp
//  TestNet
//
//  Created by Blas Eugenio Vicco on 11/29/16.
//  Copyright © 2016 Blas Eugenio Vicco. All rights reserved.
//

#include <iostream>
#include <iomanip>
#include <sstream>

#include "net.h"
#include "activation.h"

using namespace std;

const unsigned int TYPE = TANH;//SIGMOID; //TANH
const bool RBMPRETRAINING = true; //swap from false to true to activate the pre training
const double MINERRORACCEPTED = (TYPE == SIGMOID) ? 0.03 : 0.003;
const double LEARNINGRATERBM = 0.8;
const double LEARNINGRATEMLP = 0.3;
const double MOMENTUM = 0.2;

double s() {
    return TYPE == SIGMOID ? Random::onoff() : Random::sign();
}

double xxor(vector<double> i) {
    if (i[0] != i[1]) return 1.0;
    return (TYPE == SIGMOID ? 0.0 : -1.0);
}

int main(int argc, const char * argv[]) {
    cout << fixed << setprecision(3);
    int avg = 0.0;
    int numEpoch = 50;
    double avgError = 0.0;
    vector<double> inputs;
    vector<double> wish;
    classNet net = *new classNet();
    //for each epoch
    for (int t = 0; t < numEpoch; t++) {
        int c = 0;
        double gError = 0.0;
        vector<unsigned int> nppl = {6, 4, 2};
        //NETWORK SETTING
        //INPUT, OUTPUT, NUMBER PERCEPTRON PER LAYER, ACTIVATION FUNCTION, LEARNING RATE
        net.ini(2, 1, nppl, TYPE, LEARNINGRATERBM, MOMENTUM);
        
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
        
        //MLP -> Multi layer perceptron
        net.setLearningRate(LEARNINGRATEMLP);
        do {
            inputs = {s(), s()};
            wish = {xxor(inputs)};
            net.setInput(inputs);
            vector<double> output = net.getOutput();
            vector<double> error = {wish[0] - output[0]};
            gError = pow(pow(error[0], 2), 0.5);
            //UNCOMMENT TO GET LEARNING DETAILS
            //cout << "Input: " << inputs[0] << " " << inputs[1];
            //cout << " Expected: " << wish[0] << " Output: " << output[0];
            //cout << " Error: " << gError << endl; //this should be less each iteration
            net.backFix(error);
            c++;
        } while ((c < 20000) && (gError > MINERRORACCEPTED));
        cout << endl;
        cout << "||||||||||||||||||||||||||||||||||||||||||" << endl;
        if (c < 20000) {
            cout << "Min error reached at iteration: " << c << endl;
        } else {
            cout << "Max iteration reached: " << c << endl;
        }
        avg += c;
        //testing trained network
        cout << "Testing ";
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
            //cout << inputs[0] << " " << inputs[1] << " -> " << output[0] << ((gError != 0) ? " BAD!!!" : " GOD") << endl;
        }
        cout << "(Errors/Samples): " << cwrong << "/" << ctest << endl;
        cout << "||||||||||||||||||||||||||||||||||||||||||" << endl;
        cout << endl;
        avgError += (double)cwrong/(double)ctest;
    }
    cout << "----------------------------------------" << endl;
    cout << "Min error reached at iteration avg: " << avg/numEpoch << endl;
    cout << "Average error: " << 100.0 * avgError/(double)numEpoch << "%" << endl;
    cout << "----------------------------------------" << endl;
    return 0;
}
