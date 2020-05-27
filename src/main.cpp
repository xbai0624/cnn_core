#include <iostream>
#include "Tools.h"
#include "Network.h"

using namespace std;

int main(int argc, char* argv[])
{
    Network *net_work = new Network();
    net_work->Init();

    net_work->Train();

    cout<<"MAIN TEST SUCCESS!!!"<<endl;
    return 0;
}
