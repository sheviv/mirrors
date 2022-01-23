

// Передача по ссылке
void func(int &x) // x - это переменная-ссылка
{
    x = x + 1;  // приведут к изменениям исходного значения
}

// Передача по константной ссылке
void boo(const int &y) // y - это константная ссылка
{
    y = 8; // ошибка компиляции: константная ссылка не может изменить свое же значение!
}

// Передача аргументов по адресу
void boo(int *ptr)
{
    *ptr = 7;
}
int main()
{
    int value = 4;
    boo(&value);
    std::cout << "value = " << value << '\n';  // 7
    return 0;
}

// Передача по константному адресу
#include <iostream>
void printArray(const int *array, int length)
{
    // Если пользователь передал нулевой указатель в качестве array
    if (!array)
        return;
    for (int index=0; index < length; ++index)
        std::cout << array[index] << ' ';
}
int main()
{
    int array[7] = { 9, 8, 6, 4, 3, 2, 1 };
    printArray(array, 7);
}

// Передача адресов по ссылке
// tempPtr теперь является ссылкой на указатель, поэтому любые изменения tempPtr приведут и к изменениям исходного аргумента!
void setToNull(int *&tempPtr)
{
    tempPtr = nullptr; // используйте 0, если не поддерживается C++11
}
int main()
{
    // Сначала мы присваиваем ptr адрес six, т.е. *ptr = 6
    int six = 6;
    int *ptr = &six;
    // Здесь выведется 6
    std::cout << *ptr;
    // tempPtr является ссылкой на ptr
    setToNull(ptr);
    // ptr было присвоено значение nullptr!
    if (ptr)
        std::cout << *ptr;
    else
        std::cout << " ptr is null";
    return 0;
}


////////
// Возврат значений по ссылке, по адресу и по значению
// Возврат по значению
int doubleValue(int a)
{
    int value = a * 3;
    return value; // копия value возвращается здесь
} // value выходит из области видимости здесь


// Возврат по адресу
int* doubleValue(int a)
{
    int value = a * 3;
    return &value; // value возвращается по адресу здесь
} // value уничтожается здесь


// Возврат по ссылке
int& doubleValue(int a)
{
    int value = a * 3;
    return value; // value возвращается по ссылке здесь
} // value уничтожается здесь


//////////
// Перегрузка функций(несколько функций с одним и тем же именем, но с разными параметрами)
int subtract(int a, int b); // целочисленная версия
double subtract(double a, double b); // версия типа с плавающей запятой


// Параметры по умолчанию
// Правила написания
void printValue(int a=5, int b); // не правильно
void printValue(int a, int b=5); // правильно
//
void printValues(int a, int b=5)
{
    std::cout << "a: " << a << '\n';
    std::cout << "b: " << b << '\n';
}
printValues(1); // в качестве b будет использоваться значение по умолчанию - 5
printValues(6, 7); // в качестве b будет использоваться значение, предоставляемое пользователем - 7

// Несколько параметров по умолчанию
void printValues(int a=10, int b=11, int c=12)
{
    std::cout << "Values: " << a << " " << b << " " << c << '\n';
}

