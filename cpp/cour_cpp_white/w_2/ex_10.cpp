#include <iostream>
#include <string>
#include <map>

using namespace std;

map<char, int> BuildCharCounters(string& s)
{
    map<char, int> dct;
    for (const char& i : s)
    {
        ++dct[i];
    }
    return dct;
}

int main() {
    int n;
    cin >> n;
    for (int i = 0; i < n; ++i)
    {
        string f, s;
        cin >> f >> s;
        map<char, int> ff = BuildCharCounters(f);
        map<char, int> ss = BuildCharCounters(s);
        if (ff == ss)
        {
            cout << "YES" << endl;
        }
        else
        {
            cout << "NO" << endl;
        }
    }
    return 0;
}