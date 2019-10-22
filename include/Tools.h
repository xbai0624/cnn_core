#ifndef TOOLS_H
#define TOOLS_H

// here declares some commonly used routines

#include <vector>

#include <algorithm> // std::shuffle
#include <random> //std::default_random_engine
#include <chrono> // std::chrono::sytem_clock

template<class T>
void Shuffle(std::vector<T> &vec) 
{
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    shuffle(vec.begin(), vec.end(), std::default_random_engine(seed));
}


#endif
