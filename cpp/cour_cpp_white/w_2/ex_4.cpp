#include <bits/stdc++.h>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

using namespace std;

void UpdateIfGreater(int a, int& b) {
    if (a > b) {
        b = a;
    }
}

// int main()
// {
//     int first;
//     int second;
//     cin >> first;
//     cin >> second;
//     UpdateIfGreater(first, second);
//     return 0;
// }