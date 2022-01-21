#include <iostream>
int main()
{
    // Переменная choice должна быть объявлена вне цикла do while
    int choice;
    do
    {
        std::cout << "Please make a selection: \n";
        std::cout << "1) Addition\n";
        std::cout << "2) Subtraction\n";
        std::cout << "3) Multiplication\n";
        std::cout << "4) Division\n";
        std::cin >> choice;
    }
    while (choice != 1 && choice != 2 && choice != 3 && choice != 4);
        // Что-то делаем с переменной choice, например, используем оператор switch
    std::cout << "You selected option #" << choice << "\n";
    return 0;
}

///////

#include <iostream>
int main()
{
    int count(0);
    do
    {
        if (count == 5)
            continue; // переходим в конец тела цикла
        std::cout << count << " ";
        // Точка выполнения после оператора continue перемещается сюда
    } while (++count < 10); // этот код выполняется, так как он находится вне тела цикла
    return 0;
} // 0 1 2 3 4 6 7 8 9

////////

