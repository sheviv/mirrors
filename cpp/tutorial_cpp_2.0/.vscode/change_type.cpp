// для конвертации одного фундаментального типа данных в другой:
#include <iostream>
using namespace std;
int main()
{
    int i1 = 11;
    int i2 = 3;
    float x = static_cast<float>(i1) / i2;
    cout << "x: " << x << endl;
    return 0;
}

