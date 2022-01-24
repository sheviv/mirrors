#include <cassert> // для assert()
#include <array>
using namespace std;
int getArrayValue(const std::array<int, 4> &array, int index)
{
    // Предполагается, что значение index-а находится между 0 и 8
    assert(index >= 0 && index <= 8); // это строка 6 в Program.cpp
    return array[index];
}
int main()
{   
    array<int, 4> myarray = { 8, 6, 4, 1 };
    getArrayValue(myarray, -3);
    return 0;
}  // Assertion `index >= 0 && index <= 8' failed.


////////

#define NDEBUG
// Все стейтменты assert будут проигнорированы аж до самого конца этого файла

////////

// static_assert - во время компиляции, вызывая ошибку компилятора, если условие не является истинным
static_assert(sizeof(long) == 8, "long must be 8 bytes");
static_assert(sizeof(int) == 4, "int must be 4 bytes");
int main()
{
    return 0;
}

///////

