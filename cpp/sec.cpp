// #include <iostream>
// #include <stdio.h> 
// #include <cmath>
// #include <iomanip>
// #include <limits>
// #include <vector>
// #include <algorithm> 

#include <bits/stdc++.h>
using namespace std;
int main() {
    int n;
    cin >> n;
    string a;
    string need, where;
    cin.get();
    map<string, vector<string> > m;
    for (int i = 0;i < n;i++) {
        int j = 0;
        getline(cin, a);
        while(isalpha(a[j]))
           where+=a[j], j++;
           j += 3;
           for (j;j < a.size();j++) {
               while (j < a.size() &&isalpha(a[j])) {
                   need += a[j],j++;
               }
               j++;
               m[need].push_back(where);
               need.clear();
           }
           where.clear();
    }
    cout << m.size()<<endl;
    for (auto i:m) {
        cout << i.first << " - ";
        int c = i.second.size();
        for (int j = 0;j < c;j++) {
            cout << i.second[j];
            if (j + 1 < c)
                cout << ", ";
        }
        cout << endl;
    }
    return 0;
}