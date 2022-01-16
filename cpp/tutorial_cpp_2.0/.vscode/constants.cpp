#include <iostream>
#include "constants.h"

using namespace std;

int main()
{   
    // Алгоритм использования символьных констант
    int radius = 3;
    double circumference = 2 * radius * constants::pi;
    cout << circumference << endl;
}