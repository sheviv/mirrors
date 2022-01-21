#include <iostream>
#include <string>
int main()
{
    std::cout << "Enter your full name: ";
    std::string myName;
    std::getline(std::cin, myName); // полностью извлекаем строку в переменную myName
    std::cout << "Enter your age: ";
    std::string myAge;
    std::getline(std::cin, myAge); // полностью извлекаем строку в переменную myAge
    std::cout << "Your name is " << myName << " and your age is " << myAge;
}

////////

std::cin.ignore(32767, '\n'); // игнорируем символы перевода строки "\n" во входящем потоке длиной 32767 символов


////////

// Объявляем новое перечисление Colors
enum Colors
{
    // Ниже находятся перечислители - все возможные значения этого типа данных
    // Каждый перечислитель отделяется запятой (НЕ точкой с запятой)
    COLOR_RED,
    COLOR_BROWN,
    COLOR_GRAY,
    COLOR_WHITE,
    COLOR_PINK,
    COLOR_ORANGE,
    COLOR_BLUE,
    COLOR_PURPLE, // о конечной запятой читайте ниже
}; // однако сам enum должен заканчиваться точкой с запятой
// Определяем несколько переменных перечисляемого типа Colors
Colors paint = COLOR_RED;
Colors house(COLOR_GRAY);

////////

// Значения перечислителей
#include <iostream>
enum Colors
{
    COLOR_YELLOW, // присваивается 0
    COLOR_WHITE, // присваивается 1
    COLOR_ORANGE, // присваивается 2
    COLOR_GREEN, // присваивается 3
    COLOR_RED, // присваивается 4
    COLOR_GRAY, // присваивается 5
    COLOR_PURPLE, // присваивается 6
    COLOR_BROWN // присваивается 7
};
int main()
{
    Colors paint(COLOR_RED);
    std::cout << paint; // 4
    return 0;
}

/////////

// Определяем новый перечисляемый тип Animals
enum Animals
{
    ANIMAL_PIG = -4,
    ANIMAL_LION, // присваивается -3
    ANIMAL_CAT, // присваивается -2
    ANIMAL_HORSE = 6,
    ANIMAL_ZEBRA = 6, // имеет то же значение, что и ANIMAL_HORSE
    ANIMAL_COW // присваивается 7
};

/////////

enum ParseResult
{
    SUCCESS = 0,
    ERROR_OPENING_FILE = -1,
    ERROR_PARSING_FILE = -2,
    ERROR_READING_FILE = -3
};
ParseResult readFileContents()
{
    if (!openFile())
        return ERROR_OPENING_FILE;
    if (!parseFile())
        return ERROR_PARSING_FILE;
    if (!readfile())
        return ERROR_READING_FILE;
    return SUCCESS; // если всё прошло успешно
}

/////////

// Псевдонимы типов: typedef и type alias
typedef double time_t; // используем time_t в качестве псевдонима для типа double
// Следующие два стейтмента эквивалентны
double howMuch;
time_t howMuch;

////////

typedef std::vector<std::pair<std::string, int>> pairlist_t; // используем pairlist_t в качестве псевдонима для этого длиннющего типа данных
pairlist_t pairlist; // объявляем pairlist_t
boolean hasAttribute(pairlist_t pairlist) // используем pairlist_t в качестве типа параметра функции
{
    // Что-то делаем
}
// или
typedef double pairlist_t; // используем pairlist_t в качестве псевдонима для типа double

//////

// Структуры
struct Employee
{
    short id;
    int age;
    double salary;
};
Employee john; // создаем отдельную структуру Employee для John-а
john.id = 8; // присваиваем значение члену id структуры john
john.age = 27; // присваиваем значение члену age структуры john
john.salary = 32.17; // присваиваем значение члену salary структуры john

///////

struct Employee
{
    short id;
    int age;
    double salary;
};
Employee john = { 5, 27, 45000.0 }; // john.id = 5, john.age = 27, john.salary = 45000.0

////////
// Структуры и функции
#include <iostream>
using namespace std;
struct Employee
{
    short id;
    int age;
    double salary;
};
void printValues(Employee employee)
{
    cout << "id: " << employee.id << endl;
    cout << "age: " << employee.age << endl;
    cout << "salary: " << employee.salary << endl;
}
int main()
{
    Employee john = {0, 21, 90000.0};
    printValues(john);
    return 0;
}

//////

struct Employee
{
    short id;
    int age;
    float salary;
};
struct Company
{
    Employee CEO; // Employee является структурой внутри структуры Company
    int numberOfEmployees;
};
Company myCompany = {{ 3, 35, 55000.0f }, 7 };

