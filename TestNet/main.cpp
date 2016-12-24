
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

const unsigned TYPE = SIGMOID; //TANH
const double MINERRORACCEPTED = (TYPE == SIGMOID) ? 0.03 : 0.005;

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
    int maxIter = 10;
    double avgError = 0.0;
    vector<double> inputs;
    vector<double> wish;
    classNet net = *new classNet();
    for (int t = 0; t < maxIter; t++) {
        vector<unsigned int> nppl = {3};
        //NETWORK SETTING
        //INPUT, OUTPUT, NUMBER PERPECTRON PER LAYER, ACTIVATION FUNCTION, LEARNING RATE
        net.ini(2, 1, nppl, TYPE, 0.4);
        double gError = 0.0;
        int c = 0;
        /*for (unsigned int i = 0; i < 5000; i++) {
            inputs = {s(), s()};
            wish = {xxor(inputs)};
            net.setInput(inputs);
            net.temper(1, 10);
        }*/
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
        } while (gError > MINERRORACCEPTED);
        cout << endl;
        cout << "||||||||||||||||||||||||||||||||||||||||||" << endl;
        cout << "Min error reached at iteration: " << c << endl;
        avg += c;
        //testing trained network
        cout << "Testing: ";
        unsigned int cwrong = 0;
        unsigned int ctest = 50;
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
        cout << "Errors " << cwrong << " of " << ctest << endl;
        cout << "||||||||||||||||||||||||||||||||||||||||||" << endl;
        cout << endl;
        avgError += (double)cwrong/(double)ctest;
    }
    cout << "----------------------------------------" << endl;
    cout << "Min error reached at iteration avg: " << avg/maxIter << endl;
    cout << "Average error: " << 100.0 * avgError/(double)maxIter << endl;
    cout << "----------------------------------------" << endl;
    return 0;
}
