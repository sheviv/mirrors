#include <array>

///////
std::array<int, 4> myarray; // объявляем массив типа int длиной 4

///////

std::array<int, 4> myarray = { 8, 6, 4, 1 }; // список инициализаторов
std::array<int, 4> myarray2 { 8, 6, 4, 1 }; // uniform-инициализация

///////

std::array<int, 4> myarray;
myarray = { 0, 1, 2, 3 }; // ок
myarray = { 8, 6 }; // ок, элементам 2 и 3 присвоен нуль!
myarray = { 0, 1, 3, 5, 7, 9 }; // нельзя, слишком много элементов в списке инициализаторов!

///////

std::array<int, 4> myarray { 8, 6, 4, 1 };
myarray.at(1) = 7; // элемент массива под номером 1 - корректный, присваиваем ему значение 7
myarray.at(8) = 15; // элемент массива под номером 8 - некорректный, получим ошибку

///////

// Размер и сортировка
std::array<double, 4> myarray{ 8.0, 6.4, 4.3, 1.9 };
std::cout << "length: " << myarray.size();  // 4


///////
#include <iostream>
#include <array>
void printLength(const std::array<double, 4> &myarray)
{
    std::cout << "length: " << myarray.size(); // 4
}
int main()
{
    std::array<double, 4> myarray { 8.0, 6.4, 4.3, 1.9 };
    printLength(myarray);
    return 0;
}

///////

#include <algorithm> // для std::sort
std::array<int, 5> myarray { 8, 4, 2, 7, 1 };
std::sort(myarray.begin(), myarray.end()); // сортировка массива по возрастанию
std::sort(myarray.rbegin(), myarray.rend()); // сортировка массива по убыванию

///////

