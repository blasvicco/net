//
//  main.cpp
//  TestNet
//
//  Created by Blas Eugenio Vicco on 11/29/16.
//  Copyright © 2016 Blas Eugenio Vicco. All rights reserved.
//
#include <iostream>
#include "net.h"

using namespace std;

int main(int argc, const char * argv[]) {
    vector<int> nppl = {2, 4, 3};
    classNet net = *new classNet();
    net.ini(2, 2, nppl, 0.4);
    int c = 0;
    vector<float> inputs = {1, -1}; //change this to test
    vector<float> wish = {1, 1}; // change this to test
    do {
        net.setInputs(inputs);
        vector<float> output = net.getOutput();
        vector<float> errors = {wish[0] - output[0], wish[1] - output[1]};
        cout << "Output: " << output[0] << " " << output[1] << endl;
        cout << "Error: " << errors[0] << " " << errors[1]  << endl; //this should be less each iteration
        net.fix(errors);
        c++;
    } while (c < 1000);
    
    return 0;
}
