#include <iostream>
#include <string>
#include <map>
#include <set>

using namespace std;

int main() {
    int n;
    cin >> n;
    map<string, set<string>> syn_dct;

    for (int i = 0; i < n; ++i)
    {
        string mode;
        cin >> mode;

        if (mode == "ADD")
        {
            string ff, ss;
            cin >> ff >> ss;
            syn_dct[ff].insert(ss);
            syn_dct[ss].insert(ff);
        }
        else if (mode == "COUNT")
        {
            string str;
            cin >> str;
            cout << syn_dct[str].size() << endl;
        }
        else if (mode == "CHECK")
        {
            string ff, ss;
            cin >> ff >> ss;
            if (syn_dct[ff].count(ss) == 1)
            {
                cout << "YES" << endl;
            } else {
                cout << "NO" << endl;
            }
        }
    }
    return 0;
}