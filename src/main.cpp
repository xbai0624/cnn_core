#include <iostream>

#include "Tools.h"
#include "Network.h"

#include "UnitTest.h"

using namespace std;

int main(int argc, char* argv[])
{
/*
    Network *net_work = new Network();
    net_work->Init();

    net_work->Train();
*/

    // test
    UnitTest *test = new UnitTest();
    test->Test();

    cout<<"MAIN TEST SUCCESS!!!"<<endl;
    return 0;
}
