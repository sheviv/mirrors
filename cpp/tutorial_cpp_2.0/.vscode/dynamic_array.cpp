/////////
// Динамические массивы
#include <iostream>
int main()
{
    std::cout << "Enter a positive integer: ";
    int length;
    std::cin >> length;
    int *array = new int[length]; // используем оператор new[] для выделения массива. Обратите внимание, переменная length не обязательно должна быть константой!
    std::cout << "I just allocated an array of integers of length " << length << '\n';
    array[0] = 7; // присваиваем элементу под индексом 0 значение 7
    delete[] array; // используем оператор delete[] для освобождения выделенной массиву памяти
    array = 0; // используйте nullptr вместо 0 в C++11
    return 0;
}

////////

// Инициализация динамических массивов
// 0
int *array = new int[length]();

////////

// инициализации динамических массивов через списки инициализаторов:
int fixedArray[5] = { 9, 7, 5, 3, 1 }; // инициализируем фиксированный массив
int *array = new int[5] { 9, 7, 5, 3, 1 }; // инициализируем динамический массив

////////

