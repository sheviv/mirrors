// example struct()
#include <iostream>
// struct DateStruct
// {
//     int day;
//     int month;
//     int year;
// };
// void print(DateStruct &date)
// {
//     std::cout << date.day<< "/" << date.month << "/" << date.year;
// }
// int main()
// {
//     DateStruct today { 12, 11, 2018}; // используем uniform-инициализацию
//     today.day = 18; // используем оператор выбора члена для выбора члена структуры
//     print(today);  // 18/11/2018
//     std::cout << std::endl;
//     return 0;
// }


////////

// идентичны по функционалу:
// struct DateStruct
// {
//     int day;
//     int month;
//     int year;
// };
// class DateClass
// {
//     public:
//         int m_day;
//         int m_month;
//         int m_year;
// };

////////


// Методы классов
// class DateClass
// {
//     public:
//         int m_day;
//         int m_month;
//         int m_year;
//     void print() // определяем функцию-член
//     {
//         std::cout << m_day << "/" << m_month << "/" << m_year;  // 18/11/2018
//     }
// };
// int main()
// {
//     DateClass today {12, 11, 2018};
//     today.m_day = 18;  // используем оператор выбора членов для выбора переменной-члена m_day объекта today класса DateClass
//     today.print();  // используем оператор выбора членов для вызова метода print() объекта today класса DateClass
//     std::cout << std::endl;
//     return 0;
// }
// or
// #include <iostream>
#include <string>
// class Employee
// {
//     public:
//         std::string m_name;
//         int m_id;
//         double m_wage;
//     // Метод вывода информации о работнике на экран
//     void print()
//     {
//         std::cout << "Name: " << m_name <<
//         "\nId: " << m_id <<
//         "\nWage: $" << m_wage << '\n';
//     }
// };
// int main()
// {
//     // Определяем двух работников
//     Employee john { "John", 5, 30.00 };
//     Employee max { "Max", 6, 32.75 };
//     // Выводим информацию о работниках на экран
//     john.print();
//     std::cout<<std::endl;
//     max.print();
//     return 0;
// }



//////////
// public и private
// 3 уровня доступа:
// public - делает члены открытыми;
// private - делает члены закрытыми;
// protected - открывает доступ к членам только для дружественных и дочерних классов
// 
// геттеры — это функции, которые возвращают значения закрытых переменных-членов класса;
// сеттеры — это функции, которые позволяют присваивать значения закрытым переменным-членам класса.
// 
// class Date
// {
//     private:
//         int m_day;
//         int m_month;
//         int m_year;
//     public:
//         int getDay() { return m_day; } // геттер для day
//         void setDay(int day) { m_day = day; } // сеттер для day

//         int getMonth() { return m_month; } // геттер для month
//         void setMonth(int month) { m_month = month; } // сеттер для month

//         int getYear() { return m_year; } // геттер для year
//         void setYear(int year) { m_year = year; } // сеттер для year
// };



////////
// Конструкторы

// Конструкторы по умолчанию
#include <iostream>
// class Fraction
// {
//     private:
//         int m_numerator;
//         int m_denominator;
//     public:
//         Fraction() // конструктор по умолчанию
//         {
//             m_numerator = 0;
//             m_denominator = 1;
//         }  // будет вызываться, если не предоставить значения, и конструктор с параметрами, который будет вызываться, если предоставить значения.
//     int getNumerator() { return m_numerator; }
//     int getDenominator() { return m_denominator; }
//     double getValue() { return static_cast<double>(m_numerator) / m_denominator; }
// };
// int main()
// {
//     Fraction drob; // так как нет никаких аргументов, то вызывается конструктор по умолчанию Fraction()
//     std::cout << drob.getNumerator() << "/" << drob.getDenominator() << '\n';
//     return 0;
// }

// Fraction и два конструктора (по умолчанию и с параметрами)
// class Date
// {
//     private:
//         int m_day = 12;
//         int m_month = 1;
//         int m_year = 2018;
//         // Не было предоставлено конструктора, поэтому C++ автоматически создаст открытый конструктор по умолчанию
// };
// int main()
// {
//     Date date; // вызов неявного конструктора
//     return 0;
// }

// Если класс имеет другие конструкторы, то неявно генерируемый конструктор создаваться не будет.
// class Date
// {
//     private:
//         int m_day = 12;
//         int m_month = 1;
//         int m_year = 2018;
//     public:
//         Date(int day, int month, int year) // обычный конструктор (не по умолчанию)
//         {
//             m_day = day;
//             m_month = month;
//             m_year = year;
//         }
//         // Неявный конструктор не создастся, так как мы уже определили свой конструктор
// };
// int main()
// {
//     // Date date; // ошибка: Невозможно создать объект, так как конструктор по умолчанию не существует, и компилятор не сгенерировал неявный конструктор автоматически
//     Date today(14, 10, 2020); // инициализируем объект today
//     return 0;
// }


// Классы, содержащие другие классы
// class A
// {
//     public:
//         A() { std::cout << "A\n"; }
// };
// class B
// {
//     private:
//         A m_a; // B содержит A, как переменную-член
//     public:
//         B() { std::cout << "B\n"; }
// };
// int main()
// {
//     B b;
//     return 0;
// } 
// A т.к. вдруг m_a(класс А) будет использоваться в коде
// B т.к. потом переходим к вызову В()


///
// Список инициализации членов класса
// class Values
// {
//     private:
//         int m_value1;
//         double m_value2;
//         char m_value3;
//     public:
//         Values(): m_value1(3), m_value2(4.5), m_value3('d') // напрямую инициализируем переменные-члены класса
//         {
//         // Нет необходимости использовать присваивание
//         }
//         void print()
//         {
//             std::cout << "Values(" << m_value1 << ", " << m_value2 << ", " << m_value3 << ")\n";
//         }
// };
// int main()
// {
//     Values value;
//     value.print();
//     return 0;
// }

///
// добавить возможность caller-у передавать значения для инициализации
// class Values
// {
//     private:
//         int m_value1;
//         double m_value2;
//         char m_value3;
//     public:
//         Values(int value1, double value2, char value3='d')
//         : m_value1(value1), m_value2(value2), m_value3(value3) // напрямую инициализируем переменные-члены класса
//         {
//         // Нет необходимости использовать присваивание
//         }
//         void print()
//         {
//             std::cout << "Values(" << m_value1 << ", " << m_value2 << ", " << m_value3 << ")\n";
//         }
// };
// int main()
// {
//     Values value(3, 4.5); // value1 = 3, value2 = 4.5, value3 = 'd' (значение по умолчанию)
//     value.print();  // Values(3, 4.5, d)
//     return 0;
// }

///
// Инициализация переменных-членов, которые являются классами
// class A
// {
//     public:
//         A(int a) { std::cout << "A " << a << "\n"; }
// };
// class B
// {
//     private:
//         A m_a;
//     public:
//         B(int b) : m_a(b -1) // вызывается конструктор A(int) для инициализации члена m_a
//         {
//             std::cout << "B " << b << "\n";
//         }
// };
// int main()
// {
//     B b(7);
//     return 0;  // A 6 \n B 7
// }

///
// Использование списков инициализации
// class Something
// {
//     private:
//         double m_length = 3.5;
//         double m_width = 3.5;
//     public:
//         Something(double length, double width) : m_length(length), m_width(width)
//         {
//         // m_length и m_width инициализируются конструктором (значения по умолчанию, приведенные выше, не используются)
//         }
//         void print()
//         {
//             std::cout << "length: " << m_length << " and width: " << m_width << '\n';
//         }
// };
// int main()
// {
//     Something a(4.5, 5.5);
//     a.print();
//     return 0;  // length: 4.5 and width: 5.5
// }


///////
// Делегирующие конструкторы
// class Employee
// {
//     private:
//         int m_id;
//         std::string m_name;
//     public:
//         Employee(int id=0, const std::string &name=""): m_id(id), m_name(name)
//         {
//             std::cout << "Employee " << m_name << " created.\n";
//         }
//         // Используем делегирующие конструкторы для сокращения дублированного кода
//         Employee(const std::string &name) : Employee(0, name) { }
// };
// int main()
// {
//     Employee a;  // Employee  created.
//     Employee b("Ivan");  // Employee Ivan created.
//     return 0;
// }


///////
// Деструктор
// деструктор должен иметь то же имя, что и класс, со знаком тильда (~) в самом начале;
// деструктор не может принимать аргументы;
// деструктор не имеет типа возврата.
// #include <cassert>
// class Massiv
// {
//     private:
//         int *m_array;
//         int m_length;
//     public:
//         Massiv(int length) // конструктор
//         {
//             assert(length > 0);
//             m_array = new int[length];
//             m_length = length;
//         }
//         ~Massiv() // деструктор
//         {
//             // Динамически удаляем массив, который выделили ранее
//             delete[] m_array ;
//         }
//         void setValue(int index, int value) { m_array[index] = value; }
//         int getValue(int index) { return m_array[index]; }
//         int getLength() { return m_length; }
// };
// int main()
// {
// Massiv arr(15); // выделяем 15 целочисленных значений
// for (int count=0; count < 15; ++count)
// arr.setValue(count, count+1);
// std::cout << "The value of element 7 is " << arr.getValue(7);
// return 0;  // The value of element 7 is 8
// } // объект arr удаляется здесь, поэтому деструктор ~Massiv() вызывается тоже здесь


///
// Выполнение конструкторов и деструкторов
// class Another
// {
//     private:
//         int m_nID;
//     public:
//         Another(int nID)
//         {
//             std::cout << "Constructing Another " << nID << '\n';
//             m_nID = nID;
//         }
//         ~Another()
//         {
//             std::cout << "Destructing Another " << m_nID << '\n';
//         }
//         int getID() { return m_nID; }
// };
// int main()
// {
//     // Выделяем объект класса Another из стека
//     Another object(1);
//     std::cout << object.getID() << '\n';
//     // Выделяем объект класса Another динамически из кучи
//     Another *pObject = new Another(2);  // не надо
//     std::cout << pObject->getID() << '\n';
//     delete pObject;
//     return 0;
// } // объект object выходит из области видимости здесь
// Constructing Another 1
// 1
// Constructing Another 2
// 2
// Destructing Another 2  // т.к. удалили pObject до завершения выполнения функции main()
// Destructing Another 1



////////
// Скрытый указатель *this
// *this — это скрытый константный указатель, содержащий адрес объекта, который вызывает метод класса
// int main()
// {
//     Another X(3); // *this = &X внутри конструктора Another
//     Another Y(4); // *this = &Y внутри конструктора Another
//     X.setNumber(5); // *this = &X внутри метода setNumber
//     Y.setNumber(6); // *this = &Y внутри метода setNumber
//     return 0;
// }

///
// class Something
// {
//     private:
//         int data;
//     public:
//         Something(int data)
//         {
//             this->data = data;
//         }
// };


///
// Цепочки методов класса
// class Mathem
// {
//     private:
//         int m_value;
//     public:
//         Mathem() { m_value = 0; }
//         void add(int value) { m_value += value; }
//         void sub(int value) { m_value -= value; }
//         void multiply(int value) { m_value *= value; }
//         int getValue() { return m_value; }
// };
// int main()
// {
//     Mathem operation;
//     operation.add(7); // возвращает void
//     operation.sub(5); // возвращает void
//     operation.multiply(3); // возвращает void
//     std::cout << operation.getValue() << '\n';  // вернет 6 после всех операций
//     return 0;
// }

// or

// class Mathem
// {
//     private:
//         int m_value;
//     public:
//         Mathem() { m_value = 0; }
//         Mathem& add(int value) { m_value += value; return *this; }
//         Mathem& sub(int value) { m_value -= value; return *this; }
//         Mathem& multiply(int value) { m_value *= value; return *this; }
//         int getValue() { return m_value; }
// };
// int main()
// {
//     Mathem operation;
//     operation.add(7); // возвращает void
//     operation.sub(5); // возвращает void
//     operation.multiply(3); // возвращает void
//     std::cout << operation.getValue() << '\n';  // вернет 6 после всех операций
//     return 0;
// }

// or

// class Mathem
// {
//     private:
//         int m_value;
//     public:
//         Mathem() { m_value = 0; }
//         Mathem& add(int value) { m_value += value; return *this; }
//         Mathem& sub(int value) { m_value -= value; return *this; }
//         Mathem& multiply(int value) { m_value *= value; return *this; }
//         int getValue() { return m_value; }
// };
// int main()
// {
//     Mathem operation;
//     operation.add(7).sub(5).multiply(3);
//     std::cout << operation.getValue() << '\n';
//     return 0;
// }



///////
// Классы и заголовочные файлы
// Объявление методов класса вне класса
// class Mathem
// {
//     private:
//         int m_value = 0;
//     public:
//         Mathem(int value=0);
//         Mathem& add(int value);
//         Mathem& sub(int value);
//         Mathem& divide(int value);
//         int getValue() { return m_value; }
// };
// Mathem::Mathem(int value): m_value(value)
// {
// }
// Mathem& Mathem::add(int value)
// {
//     m_value += value;
//     return *this;
// }
// Mathem& Mathem::sub(int value)
// {
//     m_value -= value;
//     return *this;
// }
// Mathem& Mathem::divide(int value)
// {
//     m_value /= value;
//     return *this;
// }


// Разбить класс на файлы, для исползования вне тела функции
// Date.h:
// #ifndef DATE_H
// #define DATE_H
// class Date
// {
//     private:
//         int m_day;
//         int m_month;
//         int m_year;
//     public:
//         Date(int day, int month, int year);
//         void SetDate(int day, int month, int year);
//         int getDay() { return m_day; }
//         int getMonth() { return m_month; }
//         int getYear() { return m_year; }
// };
// #endif
// // Date.cpp:
// #include "Date.h"
// // Конструктор класса Date
// Date::Date(int day, int month, int year)
// {
//     SetDate(day, month, year);
// }
// // Метод класса Date
// void Date::SetDate(int day, int month, int year)
// {
//     m_day = day;
//     m_month = month;
//     m_year = year;
// }


/////////
// Классы и const
// Константные объекты классов
// class Anything
// {
//     public:
//         int m_value;
//         Anything(): m_value(0) { }
//         void setValue(int value) { m_value = value; }
//         int getValue() { return m_value ; }
// };
// int main()
// {
//     const Anything anything; // вызываем конструктор по умолчанию
//     anything.m_value = 7; // ошибка компиляции: нарушение const
//     anything.setValue(7); // ошибка компиляции: нарушение const
//     return 0;
// }

///
// int main()
// {
//     Anything anything;
//     anything.getValue() = "Hello!"; // вызывается неконстантный getValue()
//     const Anything anything2;
//     anything2.getValue(); // вызывается константный getValue()
//     return 0;
// }




//////////
// Статические переменные-члены класса
// int generateID()
// {
//     static int s_id = 0;
//     return ++s_id;
// }
// int main()
// {
//     std::cout << generateID() << '\n';  // 1
//     std::cout << generateID() << '\n';  // 2
//     std::cout << generateID() << '\n';  // 3
//     return 0;
// }



////////
// Дружественные функции и классы
// class Anything
// {
//     private:
//         int m_value;
//     public:
//         Anything() { m_value = 0; }
//         void add(int value) { m_value += value; }
//         // Делаем функцию reset() дружественной классу Anything
//         friend void reset(Anything &anything);
// };
// // Функция reset() теперь является другом класса Anything
// void reset(Anything &anything)
// {
//     // И мы имеем доступ к закрытым членам объектов класса Anything
//     anything.m_value = 0;
// }
// int main()
// {
//     Anything one;
//     one.add(4); // добавляем 4 к m_value
//     reset(one); // сбрасываем m_value в 0
//     return 0;
// }

///
class Humidity;
class Temperature
{
    private:
        int m_temp;
    public:
        Temperature(int temp=0) { m_temp = temp; }
        friend void outWeather(const Temperature &temperature, const Humidity &humidity);
};
class Humidity
{
    private:
        int m_humidity;
    public:
        Humidity(int humidity=0) { m_humidity = humidity; }
        friend void outWeather(const Temperature &temperature, const Humidity &humidity);
};
void outWeather(const Temperature &temperature, const Humidity &humidity)
{
    std::cout << "The temperature is " << temperature.m_temp <<
    " and the humidity is " << humidity.m_humidity << '\n';
}
int main()
{
    Temperature temp(15);
    Humidity hum(11);
    outWeather(temp, hum);
    return 0;
}