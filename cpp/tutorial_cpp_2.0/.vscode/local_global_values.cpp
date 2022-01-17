#include <iostream>
int value(4); // глобальная переменная
int main()
{
    int value = 8; // эта переменная (локальная) скрывает значение глобальной переменной
    value++; // увеличивается локальная переменная value (не глобальная)
    ::value--; // уменьшается глобальная переменная value (не локальная)
    std::cout << "Global value: " << ::value << "\n";
    std::cout << "Local value: " << value << "\n";
    return 0;
} // локальная переменная уничтожается


////////


#include <iostream>
static int g_x; // g_x - это статическая глобальная переменная, которую можно использовать только внутри этого файла
int main()
{
    return 0;
}


////////


#include <iostream>
extern double g_y(9.8); // g_y - это внешняя глобальная переменная и её можно использовать и в других файлах программы
int main()
{
    return 0;
}


////////


// global.cpp:
// Определяем две глобальные переменные
int g_m; // неконстантные глобальные переменные имеют внешнюю связь по умолчанию
int g_n(3); // неконстантные глобальные переменные имеют внешнюю связь по умолчанию
// g_m и g_n можно использовать в любом месте этого файла

// main.cpp:
#include <iostream>
extern int g_m; // предварительное объявление g_m. Теперь g_m можно использовать в любом месте этого файла
int main()
{
    extern int g_n; // предварительное объявление g_n. Теперь g_n можно использовать только внутри main()
    g_m = 4;
    std::cout << g_n; // должно вывести 3
    return 0;
}


////////


// constants.cpp:
static const double g_gravity(9.8);

// main.cpp:
#include <iostream>
extern const double g_gravity; // не найдет g_gravity в constants.cpp, так как g_gravity является внутренней переменной
int main()
{
    std:: cout << g_gravity; // вызовет ошибку компиляции, так как переменная g_gravity не была определена для использования в main.cpp
    return 0;
}

/////

// Эта функция определена как static и может быть использована только внутри этого файла.
// Попытки доступа к ней через прототип функции будут безуспешными
static int add(int a, int b)
{
    return a + b;
}


///////


// constants.cpp:
namespace Constants
{
    // Фактические глобальные переменные
    extern const double pi(3.14159);
    extern const double avogadro(6.0221413e23);
    extern const double my_gravity(9.2);
}

////////

static double g_gravity (9.8); // ограничиваем доступ к переменной только на этот файл
double getGravity() // эта функция может быть экспортирована в другие файлы для доступа к глобальной переменной
{
    return g_gravity;
}

///////

// Статическая продолжительность жизни:
#include <iostream>
void incrementAndPrint()
{
    static int s_value = 1; // переменная s_value является статической
    ++s_value;
    std::cout << s_value << std::endl;
} // переменная s_value не уничтожается здесь, но становится недоступной
int main()
{
    incrementAndPrint();
    incrementAndPrint();
    incrementAndPrint();
}

////////


// boo.h:
namespace Boo
{
    // Эта версия doOperation() принадлежит пространству имен Boo
    int doOperation(int a, int b)
    {
        return a + b;
    }
}
// asd.cpp
int main()
{
    std::cout << Boo::doOperation(5, 4); // пространства имен Boo
    return 0;
}

