#include <iostream>
int main()
{
    double *ptr(0);
    if (ptr)
        std::cout << "ptr is pointing to a double value.";
    else
        std::cout << "ptr is a null pointer.";
    std::cout << std::endl;
    return 0;
}