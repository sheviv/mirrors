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
class Employee
{
    public:
        std::string m_name;
        int m_id;
        double m_wage;
    // Метод вывода информации о работнике на экран
    void print()
    {
        std::cout << "Name: " << m_name <<
        "\nId: " << m_id <<
        "\nWage: $" << m_wage << '\n';
    }
};
int main()
{
    // Определяем двух работников
    Employee john { "John", 5, 30.00 };
    Employee max { "Max", 6, 32.75 };
    // Выводим информацию о работниках на экран
    john.print();
    std::cout<<std::endl;
    max.print();
    return 0;
}



//////////
// public и private

