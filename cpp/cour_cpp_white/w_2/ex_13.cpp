#include <iostream>
#include <string>
#include <vector>
#include <map>

using namespace std;

int main() {
    int n;
    cin >> n;
    map<vector<string>, int> bus;
    for (int i = 0; i < n; ++i)
    {
        int a;
        cin >> a;
        vector<string> stop_mode(a);
        for (string& stop : stop_mode)
        {
            cin >> stop;
        }
        if (bus.count(stop_mode) == 0)
        {
            const int num = bus.size() + 1;
            bus[stop_mode] = num;
            cout << "New bus " << num << endl;
        }
        else
        {
            cout << "Already exists for " << bus[stop_mode] << endl;
        }
    }
    return 0;
}