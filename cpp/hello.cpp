#include <algorithm>
#include <initializer_list>
#include <iostream>

int main()
{   int m, mm, mmm;
    int a = 1;
    int b = 2;
    int c = 3;

    if (a >= b && a >= c)
    {
        m = a;
        if (b >= c)
        {
            mm = b;
            mmm = c;
        }
        else
        {
            mm = c;
            mmm = b;
        }
    }
  
    elif (b >= a) and (b >= c):
        largest = b
    else:
        largest = c

    std::cout << m << "\n";
}