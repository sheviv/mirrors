#include <iostream>

using namespace std;

struct Advertising
{
    short rec;
    int age;
    double salary;
};

void printValues(Advertising ad)
{
    cout << "id: " << ad.rec << endl;
    cout << "age: " << ad.age << endl;
    cout << "salary: " << ad.salary << endl;
    cout << "sum: " << (double) (ad.salary * ad.age * ad.salary) << endl;
}

int main()
{
    // Объявляем переменную структуры Advertising
    Advertising ad;
    cout << "How many ads were shown today? ";
    cin >> ad.rec;
    cout << "What percentage of users clicked on the ads? ";
    cin >> ad.age;
    cout << "What was the average earnings per click? ";
    cin >> ad.salary;
    printValues(ad);
    return 0;
}