#include <bits/stdc++.h>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>


using namespace std;

int  main()
{   
    string a;
    string b;
    vector<int> cv;
    cin >> a;
    for (int i = 0; i < a.size(); i++)
    {   
        b = a[i];
        if (b == "f")
        {
            cv.push_back(i);
        }
    }
    if (cv.size() == 1)
    {
        cout << -1 << endl;
    }
    else if (cv.size() >= 2)
    {
        cout << cv[1] << endl;
    }
    else
    {
        cout << -2 << endl;
    }
    return 0;
}
