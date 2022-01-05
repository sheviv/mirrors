#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

using namespace std;


int main() {
    int cv;
    int n = 0;
    cin >> cv;
    vector<int> vec;

    while (n < cv)
    {
        string s;
        cin >> s;
        if (s == "COME")
        {
            int k;
            cin >> k;
            int c = vec.size() + k;
            vec.resize(c, 0);
        }
        else if (s == "WORRY")
        {
            int i;
            cin >> i;
            vec[i] = 1;
        } 
        else if (s == "QUIET")
        {
            int i;
            cin >> i;
            vec[i] = 0;
        }
        else if (s == "WORRY_COUNT")
        {
            int quanity = count(begin(vec), end(vec), 1);
            cout << quanity << endl;
        }
        n += 1;
    }
    return 0;
}