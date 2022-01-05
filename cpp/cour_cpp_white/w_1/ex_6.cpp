#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>


using namespace std;

int  main()
{   
    int a;
    int b;
    vector<int> c;
    cin >> a >> b;
    for (int i = a; i <= b; ++i)
    {
        c.push_back(i);
    }
    for (auto i : c)
    {
        if (i % 2 == 0)
        {
            cout << i << " ";
        }
    }
    cout << endl;
    return 0;
}
