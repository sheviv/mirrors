#include <iostream>
#include <string>
#include <set>
#include <map>

using namespace std;

int main() {
    int n;
    cin >> n;
    map<set<string>, int> bus;
    for (int i = 0; i < n; ++i)
    {
        int w;
        cin >> w;
        set<string> st_stop;
        for (int j = 0; j < w; ++j)
        {
            string stop;
            cin >> stop;
            st_stop.insert(stop);
        }
        if (bus.count(st_stop) == 0)
        {
            const int num = bus.size() + 1;
            bus[st_stop] = num;
            cout << "New bus " << num << endl;
        }
        else
        {
            cout << "Already exists for " << bus[st_stop] << endl;
        }
    }
    return 0;
}