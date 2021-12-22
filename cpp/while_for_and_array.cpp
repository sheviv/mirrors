#include <iostream>
#include <stdio.h> 
#include <cmath>
#include <iomanip>
#include <limits>
#include <vector>

using namespace std;

int main()
{
    int n, c, ln;
    float inf = std::numeric_limits<float>::infinity();
    c = 0;
    cin >> n;
    vector <int> a(n);
    vector <int> v;
    vector <int> v2(n);
    for (int i = 0; i < n; i++)
    {
        cin >> a[i];
    }
    for (int i = 0; i < n; i++) 
    {
        if (i != 0) 
        {
            if (a[i] > 0 && a[i - 1] > 0)
            {
                v.push_back(a[i - 1]);
                v.push_back(a[i]);
            }
            else if (a[i] < 0 && a[i - 1] < 0)
            {
                v.push_back(a[i - 1]);
                v.push_back(a[i]);
            }
        }
    }
    if (v[0] <= v[1])
    {
        cout << v[0] << " " << v[1] << endl;
    }
    else
    {
        cout << v[1] << " " << v[0] << endl;
    }
}