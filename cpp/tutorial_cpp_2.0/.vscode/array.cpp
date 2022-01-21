// Даже если array является фактическим массивом, внутри этой функции он должен рассматриваться как константный
// void passArray(const int array[5])
// {
//     // Поэтому каждая из следующих строк вызовет ошибку компиляции!
//     array[0] = 11;
//     array[1] = 7;
//     array[2] = 5;
//     array[3] = 3;
//     array[4] = 2;
// }

///////

// std::cout << sizeof(array) << '\n'; // выводится размер массива

///////

// Определение длины фиксированного массива
// #include <iostream>
// int main()
// {
//     int array[] = { 1, 3, 3, 4, 5, 9, 14, 17 };
//     std::cout << "The array has: " << sizeof(array) / sizeof(array[0]) << " elements\n";
//     return 0;
// }

///////

// Поместите перечисление в пространство имен. Объявите массив, где элементами будут эти перечислители и, используя
// список инициализаторов, инициализируйте каждый элемент соответствующим количеством лап определенного животного.
// #include <iostream>
// namespace Animals
// {
//     enum Animals
//     {
//         CHICKEN,
//         LION,
//         GIRAFFE,
//         ELEPHANT,
//         DUCK,
//         SNAKE,
//         MAX_ANIMALS
//     };
// }
// int main()
// {
//     int legs[Animals::MAX_ANIMALS] = {2, 4, 4, 4, 2, 0};
//     std::cout << "ELEPHANT: " << legs[Animals::ELEPHANT] << " legs.\n";
//     return 0;
// }

///////

// int students[] = { 73, 85, 84, 44, 78};
// const int numStudents = sizeof(students) / sizeof(students[0]);
// int totalScore = 0;
// Используем цикл для вычисления totalScore
// for (int person = 0; person < numStudents; ++person)
//     totalScore += students[person];
// double averageScore = static_cast<double>(totalScore) / numStudents;

///////

// std::swap(a, b); // меняем местами значения переменных a и b

///////
// sotred min to max
// #include <iostream>
// #include <algorithm>
// using namespace std;
// int main()
// {
//     const int leight = 5;
//     int array[leight] = { 30, 50, 20, 10, 40 };
//     for (int startIndex = 0; startIndex < leight - 1; ++startIndex)
//     {
//         int smallestIndex = startIndex;
//         for (int currentIndex = startIndex + 1; currentIndex < leight; ++currentIndex)
//         {
//             if (array[currentIndex] < array[smallestIndex])
//             {   
//                 smallestIndex = currentIndex;
//             }
//         }
//         swap(array[startIndex], array[smallestIndex]);
//     }
//     for (int i = 0; i < leight; ++i)
//     {
//         cout << array[i] << " ";
//     }
//     return 0;
// }

////////

// Доступ к элементам в двумерном массиве
// for (int row = 0; row < numRows; ++row) // доступ по строкам
//     for (int col = 0; col < numCols; ++col) // доступ к каждому элементу в строке
//         std::cout << array[row][col];


////////


// double array
// multiply table
#include <iostream>
using namespace std;
int main()
{
    const int roww = 10;
    const int coll = 10;
    int array[roww][coll] = {0};

    for (int row = 0; row < roww; ++row)
    {
        for (int col = 0; col < coll; ++col)
        {
            array[row][col] = row * col;
        }
    }
    for (int i = 1; i < roww; ++i)
    {
        for (int j = 1; j < coll; ++j)
        {
            cout << array[i][j] << " ";
        }
        cout << endl;
    }
}