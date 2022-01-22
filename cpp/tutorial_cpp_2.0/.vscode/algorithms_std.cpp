#include <algorithm>
#include <array>
#include <iostream>
#include <vector>
using namespace std;

// std::find() и поиск элемента по значению
std::vector<int> v{1, 2, 3, 4};
int n1 = 3;
auto result1 = std::find(begin(v), end(v), n1);

// std::find_if() и поиск элемента с условием
auto is_even = [](int i){ return i%2 == 0; };
auto result3 = std::find_if(begin(v), end(v), is_even);

// std::count() и подсчет вхождений элемента
vector<int> vect{ 3, 2, 1, 3, 3, 5, 3 };
cout << "Number of times 3 appears : " << count(vect.begin(), vect.end(), 3);  // 4

// std::count_if() и подсчет вхождений элемента
bool IsOdd (int i) { return ((i%2)==1); }
int mycount = count_if (vect.begin(), vect.end(), IsOdd);

// std::for_each() и все элементы контейнера
void doubleNumber(int &i)
{
    i *= 2;
}
std::array arr{ 1, 2, 3, 4 };
std::for_each(arr.begin(), arr.end(), doubleNumber);
