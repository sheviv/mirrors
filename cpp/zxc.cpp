// Обработка числовых аргументов
#include <iostream>
#include <string>
#include <sstream> // для std::stringstream
#include <cstdlib> // для exit()
int main(int argc, char *argv[])
{
    if (argc <= 1)
    {
        // В некоторых операционных системах argv[0] может быть просто пустой строкой, без имени программы
        // Обрабатываем случай, когда argv[0] может быть пустым или не пустым
        if (argv[0])
            std::cout << "Usage: " << argv[0] << " <number>" << '\n';
        else
            std::cout << "Usage: <program name> <number>" << '\n';
        exit(1);
    }
    std::stringstream convert(argv[1]); // создаем переменную stringstream с именем convert, инициализируя её значением argv[1]
    int myint;
    if (!(convert >> myint)) // выполняем конвертацию
        myint = 0; // если конвертация терпит неудачу, то присваиваем myint значение по умолчанию
    std::cout << "Got integer: " << myint << '\n';
    return 0;
}
