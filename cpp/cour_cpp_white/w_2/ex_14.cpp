#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <set>

using namespace std;

// int main() {
//     int n;
//     cin >> n;
//     map<vector<string>, int> bus;
//     set<string> st;
//     for (int i = 0; i < n; ++i)
//     {
//         string a;
//         cin >> a;
//         st.insert(a);
//     }
//     cout << st.size() << endl;
//     return 0;
// }

set<string> BuildMapValuesSet(const map<int, string>& m)
{
    set<string> st;
    for (auto i : m)
    {
        st.insert(i.second);
    }
    // cout << st.size() << endl;
    return st;
}

int main()
{
    set<string> values = BuildMapValuesSet({
        {1, "odd"},
        {2, "even"},
        {3, "odd"},
        {4, "even"},
        {5, "odd"}});
    for (auto i : values)
    {
        cout << i << endl;
    }
    return 0;
}