//
//  random.hpp
//  test
//
//  Created by Blas Eugenio Vicco on 12/23/16.
//  Copyright © 2016 Blas Eugenio Vicco. All rights reserved.
//

#ifndef random_hpp
#define random_hpp

#include <time.h>
#include <random>

class Random {
public:
    static int sign();
    static int onoff();
    static double dur01();
    static int rndRange(int from, int to);
};

#endif /* random_hpp */
