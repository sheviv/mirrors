#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>


using namespace std;

int  main()
{   
    double n;
    double a;
    double b;
    double x;
    double y;
    cin >> n >> a >> b >> x >> y;
    if (n > b)
    {
        n *= (1 - y / 100);
    } else if (n > a)
    {
        n *= (1 - x / 100);
    }
    cout << n;
    return 0;
}

// #include <iostream>
// using namespace std;
// int main() {
//     double n, a, b, x, y;
//     cin >> n >> a >> b >> x >> y;
//     if (n > b) {
//         n *= (1 - y / 100);
//     } else if (n > a) {
//         n *= (1 - x / 100);
//     }
//     cout << n;
//     return 0;
// }