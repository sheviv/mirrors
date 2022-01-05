#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>


using namespace std;

int  main()
{
    int a;
    int b;
    int c;
    double d;
    double b1;
    double b2;
    // vector<string> cv;
    cin >> a >> b >> c;
    // cv.push_back(a);
    // cv.push_back(b);
    // cv.push_back(c);
    // sort(cv.begin(), cv.end());
    if (a != 0)
    {
        d = pow(b, 2) - 4 * a * c;
        if (d > 0)
        {
            b1 = (b * (-1) + sqrt(d)) / (2 * a);
            b2 = (b * (-1) - sqrt(d)) / (2 * a);
            cout << b1 << " " << b2 << endl;

        }
        else if (d == 0)
        {
            b1 = (b * (-1) + sqrt(d)) / (2 * a);
            cout << (double) b1 << endl;
        }
        else
        {
            return 0;
        }
    }
    else
    {
        if (b != 0)
        {
            cout << (double) (c * (-1)) / b << endl;
        }
        else
        {
            return 0;
        }
    }
    return 0;
}