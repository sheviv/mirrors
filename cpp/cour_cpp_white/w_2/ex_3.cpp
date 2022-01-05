#include <bits/stdc++.h>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

using namespace std;

bool IsPalindrom (string x)
{
    string p = x;
    reverse(p.begin(), p.end());
    if (p == x)
    {
        return true;
    }
    else
    {
        return false;
    }
}

int main()
{
    string n;
    cin >> n;
    if (IsPalindrom(n) == 0)
    {
        cout << boolalpha << false << endl;
    }
    else
    {
        cout << boolalpha << true << endl;
    }
    return 0;
}
