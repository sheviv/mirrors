#include <iostream>

using namespace std;

int  main()
{
    int c = 1;
    cout << "c_1: " << c << endl;
    ++c;
    cout << "c_2: " << c << endl;
    c *= 5;
    cout << "c_3: " << c << endl;
    c -= 3;
    cout << "c_4: " << c << endl;
    cout << c++ << endl;
    return 0;
}