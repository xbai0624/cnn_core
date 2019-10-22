#include <iostream>
#include "Tools.h"

using namespace std;

int main(int argc, char* argv[])
{
    cout<<"Test."<<endl;

    vector<double> vec;
    for(int i=0;i<10;i++){
        vec.push_back(0.9 * i);
    }

    for(auto&i: vec){
        cout<<i<<", ";
    }
    cout<<endl;

    Shuffle<double>(vec);

    for(auto&i: vec){
        cout<<i<<", ";
    }
    cout<<endl;

    return 0;
}
