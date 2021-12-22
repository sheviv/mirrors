#include <iostream>
#include <stdio.h> 
#include <cmath>
#include <iomanip>
#include <limits>
#include <vector>
#include <algorithm> 

using namespace std;

int main()
{
    int n, c, ln;
    float inf = std::numeric_limits<float>::infinity();
    c = 0;
    cin >> n;
    vector <int> a(n);
    vector <int> v;
    vector <int> v2;
    for (int i = 0; i < n; i++)
    {   
        cin >> a[i];
    }
    // cout << "v[0]: " << v[0] << endl;
    // for (auto now : v)
    // {
    //     cout << now << " ";
    // }
    // // cout << "v[0]: " << v << endl;

    // int num_items = std::count(a.cbegin(), a.cend(), 3);
    // cout << "num_items: " << endl;
    // cout << num_items << endl;

    for (int i = 0; i < n; i++) 
    {   
        // const int num_items = std::count(v.cbegin(), v.cend(), a[i]);
        int num_items = std::count(a.cbegin(), a.cend(), a[i]);
        if (num_items == 1)
        {
            v.push_back(a[i]);
        }
    }
    // cout << "v: " << endl;
    for (auto now : v)
    {
        cout << now << " ";
    }
    cout << endl;
}