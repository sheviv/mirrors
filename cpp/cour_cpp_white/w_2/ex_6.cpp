#include <bits/stdc++.h>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

using namespace std;

int main()
{
    int first;
    int n;
    int ss = 0;
    double sm;
    cin >> first;
    vector<int> cv;
    vector<int> asd;
    for (int i = 0; i < first; ++i)
    {
        cin >> n;
        cv.push_back(n);
    }
    for (auto i : cv)
    {
        ss += i;
    }
    sm = ss / cv.size();

    for (int i = 0; i < cv.size(); ++i)
    {   
        double df = cv[i];
        if (df > sm)
        {
            asd.push_back(i);
        }
    }

    cout << asd.size() << endl;
    for (auto j : asd)
    {
        cout << j << " ";
    }
    cout << endl;
    return 0;
}