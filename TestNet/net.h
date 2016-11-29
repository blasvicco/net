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
    vector <classLayer> Layer;
    vector <float> inputs;
    vector <float> outputs;
    unsigned long ni;
    unsigned long no;
    unsigned long nl;
    vector <int> nppl;
    int feedForward();
    
public:
    classNet();
    ~classNet();
    void ini(int ninput, int noutput, vector <int> numppl, float initialMu);
    void setInputs(vector <float> input);
    vector <float> getOutput();
    void fix(vector <float> errors);
};

#endif /* net_h */
