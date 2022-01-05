#include <iostream>
#include <vector>

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