
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

const unsigned TYPE = TANH;//SIGMOID; //TANH
const double MINERRORACCEPTED = (TYPE == SIGMOID) ? 0.01 : 0.005;

double s() {
    return rand() % 2 == 1 ? (TYPE == SIGMOID ? 0.0 : -1.0) : 1.0;
}

double xxor(vector<double> i) {
    if (i[0] != i[1]) return 1.0;
    return (TYPE == SIGMOID ? 0.0 : -1.0);
}

int main(int argc, const char * argv[]) {
    cout << fixed << setprecision(3);
    srand((unsigned int)time(NULL));
    int avg = 0.0;
    int maxIter = 10;
    vector<double> inputs;
    vector<double> wish;
    classNet net = *new classNet();
    for (int j = 0; j < maxIter; j++) {
        vector<unsigned int> nppl = {3};
        //NETWORK SETTING
        //INPUT, OUTPUT, NUMBER PERPECTRON PER LAYER, ACTIVATION FUNCTION, LEARNING RATE
        net.ini(2, 1, nppl, TYPE, 0.6);
        double gError = 0.0;
        double lGError = 0.0;
        int c = 0;
        do {
            lGError = gError;
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
        } while (gError > MINERRORACCEPTED);
        cout << endl;
        cout << "||||||||||||||||||||||||||||||" << endl;
        cout << "Min error reached at iteration: " << c << endl;
        cout << "||||||||||||||||||||||||||||||" << endl;
        cout << endl;
        avg += c;
    }
    cout << "----------------------------------------" << endl;
    cout << "Min error reached at iteration avg: " << avg/maxIter << endl;
    cout << "----------------------------------------" << endl;
    //testing trained network
    cout << "Testing: " << endl;
    for (unsigned int t = 0; t < 10; t++) {
        inputs = {s(), s()}; //change this to test
        net.setInput(inputs);
        vector<double> output = net.getOutput();
        cout << inputs[0] << " " << inputs[1] << " -> " << output[0] << endl;
    }
    return 0;
}
