//
//  net.h
//
//  Created by Blas Eugenio Vicco on 11/9/16.
//  Copyright © 2016 Blas Eugenio Vicco. All rights reserved.
//

#ifndef net_h
#define net_h

#include "layer.h"

class classNet {
private:
    vector <classLayer> Layer;//capas
    vector <double> input;//entradas
    vector <double> output;//salidas obtenidas
    unsigned int ni;//numero de entradas
    unsigned int no;//numero de salidas
    unsigned int nl;//numero de capas
    vector <unsigned int> nppl;//numero de perceptrones por capa
    vector<double> getGens();
    void setGens(vector<double> gens);
    void mutate(vector<double> gens, double mRate);
    void feedForward(unsigned int deep);
    
public:
    classNet();
    ~classNet();
    void ini(unsigned int ninput, unsigned int noutput, vector <unsigned int> numppl, unsigned int type, double initialMu, double initialMomentum);
    void clone(classNet Net, double mRate);
    vector <classLayer> getLayer();
    unsigned int getNi();
    unsigned int getNo();
    void setInput(vector <double> inp);
    void setLearningRate(double rate);
    vector <double> getOutput();
    void backFix(vector <double> error);
    void temper(unsigned int deep, unsigned int k);
};

#endif /* net_h */
