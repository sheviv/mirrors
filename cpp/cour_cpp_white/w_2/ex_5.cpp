#include <bits/stdc++.h>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

using namespace std;

// void MoveStrings(vector<string>& source, vector<string>& destination)
// {
//     for (auto j : source)
//     {
//         destination.push_back(j);
//     }
//     source.clear();
// }

// void Reverse(vector<int>& v)
// {
//     reverse(v.begin(), v.end());
// }

vector<int> Reversed(const vector<int>& v)
{
    vector<int> cop = v;
    reverse(cop.begin(), cop.end());
    return cop;
}

// int main()
// {
//     vector<string> source;
//     vector<string> destination;
//     cin >> source;
//     cin >> destination;
//     MoveStrings(source, destination);
//     for (auto i : source)
//     {
//         cout << i << " ";
//     }
//     cout << endl;
//     for (auto j : destination)
//     {
//         cout << j << " ";
//     }
//     cout << endl;
//     return 0;
// }