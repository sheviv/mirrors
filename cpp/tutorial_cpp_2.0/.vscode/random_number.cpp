// #include <iostream>
// #include <cstdlib> // для функций rand() и srand()
// int main()
// {
//     srand(4541); // устанавливаем стартовое значение - 4 541
//     // Выводим 100 случайных чисел
//     for (int count=0; count < 100; ++count)
//     {
//         std::cout << rand() << "\t";
//         // Если вывели 5 чисел, то вставляем символ новой строки
//         if ((count+1) % 5 == 0)
//             std::cout << "\n";
//     }
// }

////////

#include <iostream>
#include <cstdlib> // для функций rand() и srand()
#include <ctime> // для функции time()
int main()
{
    srand(static_cast<unsigned int>(time(0))); // устанавливаем значение системных часов в качестве стартового числа
    for (int count=0; count < 100; ++count)
    {
        std::cout << rand() << "\t";
        // Если вывели 5 чисел, то вставляем символ новой строки
        if ((count+1) % 5 == 0)
            std::cout << "\n";
    }
}

////////

// рандомное число между значениями min и max.
int getRandomNumber(int min, int max)
{
    static const double fraction = 1.0 / (static_cast<double>(RAND_MAX) + 1.0);
    // Равномерно распределяем рандомное число в нашем диапазоне
    return static_cast<int>(rand() * fraction * (max - min + 1) + min);
}

////////

