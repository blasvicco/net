//
//  layer.h
//
//  Created by Blas Eugenio Vicco on 11/9/16.
//  Copyright © 2016 Blas Eugenio Vicco. All rights reserved.
//

#ifndef layer_h
#define layer_h

#include "perceptron.h"

class classLayer {
private:
    vector <classPerceptron> Perceptron;
    unsigned int activationType;
    vector <double> input;
    vector <double> vbias; //restricted boltzman machine visible bias
    vector <double> vbiasLastUpdate;
    vector <double> output;
    vector <double> delta;
    unsigned int np;
    unsigned int ni;
    double mu = 0.6;
    double momentum;
    vector<double> feedForward(vector<double> input);
    vector<double> feedBackware(vector<double> input);
    vector<double> binomial(vector<double> input);
    
public:
    classLayer(unsigned int nip, unsigned int nperce, unsigned int type, double initialMu, double initialMomentum);
    ~classLayer();
    vector<classPerceptron> getPerceptron();
    void setInput(vector <double> inp);
    void setLearningRate(double rate);
    vector<double> getOutput();
    void setError(vector<double> error);
    vector<double> backFix();
    void temper(unsigned int k);
};

#endif /* layer_h */
