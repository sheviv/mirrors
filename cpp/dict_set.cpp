#include <iostream>
#include <set>
#include <iterator>
#include <vector>

using namespace std;

int main() {
    set <int> s;
    set <int> ss;
    int n;
    int nn;
    cin >> n;
    for (int i = 0; i < n; ++i)
    {   
        int x;
        cin >> x;
        s.insert(x);
    }
    cin >> nn;
    for (int i = 0; i < nn; ++i)
    {   
        int xx;
        cin >> xx;
        ss.insert(xx);
    }
    
    if (s.size() >= ss.size())
    {   
        // int cc = 0;
        for (auto now : s)
        {    
            if (ss.find(now) != ss.end())
            {   
                // ++cc;
                cout << now << " ";
            }
        }
        cout << endl;
        // cout << cc << endl;
    }
    else
    {   
        // int cc = 0;
        for (auto now : ss)
        {    
            if (s.find(now) != s.end())
            {   
                // ++cc;
                cout << now << " ";
            }
        }
        cout << endl;
        // cout << cc << endl;
    }
    return 0;
}