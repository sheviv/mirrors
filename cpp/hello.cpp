#include <iostream>
#include <stdio.h> 
#include <cmath>
#include <iomanip>
#include <limits>
#include <vector>

using namespace std;

int fib(int n)
{   
    int cur = 1;
    if (n > 2)
    {
        return cur = fib(n - 1) + fib(n - 2);
    }
    return cur;
}

int main()
{
    // double n;
    int n;
    cin >> n;
    // cin >> b;

    cout << fib(n) << endl;
    return 0;
}