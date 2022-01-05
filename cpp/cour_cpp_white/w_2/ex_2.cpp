#include <iostream>

using namespace std;

int Factorial (int x)
{
    if (x < 1)
    {
        return 1;
    }
    else
    {
        x = x * (Factorial((x - 1)));
        return x;
    }
}

int main()
{
    int n;
    cin >> n;
    cout << Factorial(n) << endl;
    return 0;
}
