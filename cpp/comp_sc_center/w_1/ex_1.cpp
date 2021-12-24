#include <iostream>
#include <cmath>

using namespace std;

int main()
{
    int c;
    int a;
    int b;
    double x1, x2, d;
    cin >> a >> b >> c;
    d = pow(b, 2) - 4 * a * c;
    // cout << "d: " << d << endl;
    if (d > 0)
    {   
        // cout << "d>0" << endl;
        x1 = (-1 * b + sqrt(d)) / (2 * a);
        x2 = (-1 * b - sqrt(d)) / (2 * a);
        cout << x1 << " " << x2 << endl;
    }
    else if (d < 0)
    {   
        // cout << "d<0" << endl;
        cout << "No real roots" << endl;
    }
    else
    {   
        // cout << "d=0" << endl;
        x1 = (-b) / (2 * a);
        cout << x1 << " " << x1 << endl;
    }
    return 0;
}