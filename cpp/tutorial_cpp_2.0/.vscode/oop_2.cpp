// Анонимные объекты
#include <iostream>
// int add(int a, int b)
// {
//     return a + b; // анонимный объект создается для хранения и возврата результата выражения a + b
// }
// int main()
// {
//     std::cout << add(4, 2);
//     return 0;
// }


///////

// Анонимные объекты класса
// Dollars dollars(7); // обычный объект класса
// Dollars(8); // анонимный объект класса


///////

// example
// class Dollars
// {
//     private:
//         int m_dollars;
//     public:
//         Dollars(int dollars) { m_dollars = dollars; }
//         int getDollars() const { return m_dollars; }
// };
// Dollars add(const Dollars &d1, const Dollars &d2)
// {
//     return Dollars(d1.getDollars() + d2.getDollars()); // возвращаем анонимный объект класса Dollars
// }
// int main()
// {
//     Dollars dollars1(7);
//     Dollars dollars2(9);
//     std::cout << "I have " << add(dollars1, dollars2).getDollars() << " dollars." << std::endl; // выводим анонимный объект класса Dollars
//     return 0;
// }



////////
// Вложенные типы данных в классах
// example 1
// class Fruit
// {
//     public:
//         // Мы переместили FruitList внутрь класса под спецификатор доступа public
//         enum FruitList
//         {
//             AVOCADO,
//             BLACKBERRY,
//             LEMON
//         };
//     private:
//         FruitList m_type;
//     public:
//         Fruit(FruitList type) :
//         m_type(type)
//         {
//         }
//         FruitList getType() { return m_type; }
// };
// int main()
// {
//     // Доступ к FruitList осуществляется через Fruit
//     Fruit avocado(Fruit::AVOCADO);
//     if (avocado.getType() == Fruit::AVOCADO)
//         std::cout << "I am an avocado!";
//     else
//         std::cout << "I am not an avocado!";
//     std::cout << std::endl;
//     return 0;
// }



/////////
// Измерение времени выполнения (тайминг) кода
int main()
{
    int a = 1;
    int b = 2;
    std::cout << ((a > b) ? a : b) << std::endl;
    return 0;
}