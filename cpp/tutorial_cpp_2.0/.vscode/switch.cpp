#include <iostream>
enum Colors
{
    COLOR_GRAY,
    COLOR_PINK,
    COLOR_BLUE,
    COLOR_PURPLE,
    COLOR_RED
};
void printColor(Colors color)
{
    switch (color)
    {
        case COLOR_GRAY:
            std::cout << "Gray" << std::endl;
            break;
        case COLOR_PINK:
            std::cout << "Pink" << std::endl;
            break;
        case COLOR_BLUE:
            std::cout << "Blue" << std::endl;
            break;
        case COLOR_PURPLE:
            std::cout << "Purple" << std::endl;
            break;
        case COLOR_RED:
            std::cout << "Red" << std::endl;
            break;
        default:
            std::cout << "Unknown" << std::endl;
            break;
    }
}
int main()
{
    printColor(COLOR_BLUE);
    return 0;
}

////////

bool isDigit(char p)
{
    switch (p)
    {
        case '0': // если p = 0
        case '1': // если p = 1
        case '2': // если p = 2
        case '3': // если p = 3
        case '4': // если p = 4
        case '5': // если p = 5
        case '6': // если p = 6
        case '7': // если p = 7
        case '8': // если p = 8
        case '9': // если p = 9
            return true; // возвращаем true
        default: // в противном случае, возвращаем false
            return false;
    }
}