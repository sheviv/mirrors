// binary = std::bitset<8>(n).to_string();

#include <bits/stdc++.h>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

using namespace std;

int main() {
    int n;
    vector<int> cv;
    cin >> n;

    while (n > 0)
    {
        cv.push_back(n % 2);
        n /= 2;
    }
    for (int i = cv.size() - 1; i >= 0; --i)
    {
        cout << cv[i];
    }
    return 0;
}