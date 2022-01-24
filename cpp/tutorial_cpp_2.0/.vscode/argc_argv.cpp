// Программа MyArguments
#include <iostream>
int main(int argc, char *argv[])
{
    std::cout << "There are " << argc << " arguments:\n";
    // Перебираем каждый аргумент и выводим его порядковый номер и значение
    for (int count=0; count < argc; ++count)
        std::cout << count << " " << argv[count] << '\n';
    return 0;
}
// 1. g++ -o argc_argv argc_argv.cpp
// 2. ./argc_argv 2 12 34
// 1 2
// 2 12
// 3 34


///////

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


///////

// 1-ый аргумент - кол-во переменных
// 2-ой и т.д. аргументы - переменные
#include <iostream>
#include <cstdarg> // требуется для использования эллипсиса
// Эллипсис должен быть последним параметром.
// Переменная count - это количество переданных аргументов
double findAverage(int count, ...)
{
    double sum = 0;
    // Мы получаем доступ к эллипсису через va_list, поэтому объявляем переменную этого типа
    va_list list;
    // Инициализируем va_list, используя va_start.
    // Первый параметр - это список, который нужно инициализировать.
    // Второй параметр - это последний параметр, который не является эллипсисом
    va_start(list, count);
    // Перебираем каждый из аргументов эллипсиса 
    for (int arg=0; arg < count; ++arg)
         // Используем va_arg для получения параметров из эллипсиса.
         // Первый параметр - это va_list, который мы используем.
         // Второй параметр - это ожидаемый тип параметров
         sum += va_arg(list, int);
    // Выполняем очистку va_list, когда уже сделали всё необходимое 
    va_end(list);
    return sum / count;
}
int main()
{
    std::cout << findAverage(4, 1, 2, 3, 4) << '\n';
    std::cout << findAverage(5, 1, 2, 3, 4, 5) << '\n';
}

