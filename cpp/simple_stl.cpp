#include <iostream>
#include <set>
#include <iterator>
#include <vector>
#include <algorithm>

using namespace std;

// void print(set <int> s)
void print(vector <int> s)
{
    for (auto now : s)
    {
        cout << "now: " << now << " ";
        // cout << "now: " << now << " " << endl;
    }
    cout << endl;
}

int main()
{   
    // int sc;
    int n;
    // set <int> s;
    // cin >> sc;
    cin >> n;
    vector <int> s;
    for (int i = 0; i < n; ++i)
    {   
        int x;
        cin >> x;
        // s.insert(x);
        s.push_back(x);
    }
    sort(s.begin(), s.end());
    // print(s);
    while (next_permutation(s.begin(), s.end()))
    {
        print(s);
    }
    return 0;
}