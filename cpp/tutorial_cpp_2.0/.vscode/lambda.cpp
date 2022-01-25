// example
// #include <iostream>
// int main()
// {
//     []() {}; // определяем лямбда-выражение без captureClause[], параметров и возвращаемого типа
//     return 0;
// }


///////

// #include <iostream>
// int main()
// {
//     auto greet_john_doe ([] {std::cout << "Hello, John Doe" << std::endl;
//     });
//     greet_john_doe();
// } // Hello, John Doe


///////

// #include <iostream>
// int main()
// {   
//     int ll = 5;
//     int rr = 7;
//     auto plus ([](auto l, auto r) {return l + r;});
//     std::cout << "plus: " << plus(ll, rr) << std::endl;
// } // plus


///////

// Хорошо: Мы можем хранить лямбду в именованной переменной и передавать её в функцию в качестве параметра
// auto isEven{
//     [](int i)
//     {
//         return ((i % 2) == 0);
//     }
// };
// return std::all_of(array.begin(), array.end(), isEven);


///////

// #include <iostream>
// int main()
// {
//     // Примечание: Явно указываем тип double для возвращаемого значения
//     auto divide{ [](int x, int y, bool bInteger) -> double {
//     if (bInteger)
//         return x / y; // выполнится неявное преобразование в тип double
//     else
//         return static_cast<double>(x) / y;
//     }};
//     std::cout << divide(3, 2, true) << '\n';
//     std::cout << divide(3, 2, false) << '\n';
//     return 0;
// }


///////
// лямбда-захваты - доступ к переменным из окружающей области видимости
#include <iostream>
int main()
{
    int ammo{10};
    auto shoot{
        [&ammo]() {  // передаем значение global  // cout - 9
        // or
        // [ammo]() mutable{  // передаем значение local  // cout - 10
            // Illegal, ammo was captured as a const copy.
            --ammo;
            std::cout << "Pew! " << ammo << " shot(s) left.\n";
            }};
    shoot();
    std::cout << ammo << " shot(s) left\n";
    return 0;
}


///////
#include <vector>
int health{ 33 };
int armor{ 100 };
vector<CEnemy> enemies{};
// Захватываем переменные health и armor по значению, а enemies – по ссылке
[health, armor, &enemies](){};
// Захватываем переменную enemies по ссылке, а все остальные – по значению
[=, &enemies](){};
// Захватываем переменную armor по значению, а все остальные – по ссылке
[&, armor](){};
// Запрещено, так как мы уже определили захват по ссылке для всех переменных
[&, &armor](){};
// Запрещено, так как мы уже определили захват по значению для всех переменных
[=, armor](){};
// Запрещено, так как переменная armor используется дважды
[armor, &health, &armor](){};
// Запрещено, так как захват по умолчанию должен быть первым элементом в списке захвата
[armor, &](){};

