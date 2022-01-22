// ссылки
int value = 7; // обычная переменная
int &ref = value; // ссылка на переменную value

////////

int value = 7; // обычная переменная
int &ref = value; // ссылка на переменную value
value = 8; // value теперь 8
ref = 9; // value теперь 9
std::cout << value << std::endl; // выведется 9
++ref;
std::cout << value << std::endl; // выведется 10

/////

// l-value — это объект, который имеет определенный адрес памяти
// r-value — это временное значение без определенного адреса памяти и с областью видимости выражения


// Инициализация ссылок
int value = 7;
int &ref = value; // корректная ссылка: инициализирована переменной value
int &invalidRef; // некорректная ссылка: ссылка должна ссылаться на что-либо

////////

// Ссылки в качестве параметров в функциях
#include <iostream>
// ref - это ссылка на переданный аргумент, а не копия аргумента
void changeN(int &ref)
{
    ref = 8;
}
int main()
{
    int x = 7;
    std::cout << x << '\n';
    changeN(x); // обратите внимание, этот аргумент не обязательно должен быть ссылкой
    std::cout << x << '\n';
    return 0;
}

//////////

// Оператор доступа к членам через указатель
struct Man
{
    int weight;
    double height;
};
Man man; // определяем переменную структуры Man
// Доступ к члену осуществляется через ссылку на переменную структуры Man
Man &ref = man;
ref.weight = 60;


///////

struct Man
{
    int weight;
    double height;
};
Man man;
// Доступ к члену осуществляется через указатель на переменную структуры Man
Man *ptr = &man;
(*ptr).weight = 60;
// or
ptr->weight = 60;


////////

int array[7] = { 10, 8, 6, 5, 4, 3, 1 };
for (auto &element: array) // символ амперсанда делает element ссылкой на текущий элемент массива, предотвращая копирование
    std::cout << element << ' ';


///////

// обычные ссылки или константные ссылки в качестве объявляемого элемента в цикле foreach (в целях улучшения производительности).
int array[7] = { 10, 8, 6, 5, 4, 3, 1 };
for (const auto &element: array) // element - это константная ссылка на текущий элемент массива в итерации
    std::cout << element << ' ';

///////

// «общий указатель»
int nResult;
float fResult;
struct Something
{
    int n;
    float f;
};
Something sResult;
void *ptr; // общий указатель void
ptr = &nResult; // допустимо
ptr = &fResult; // допустимо
ptr = &sResult; // допустимо


////////

