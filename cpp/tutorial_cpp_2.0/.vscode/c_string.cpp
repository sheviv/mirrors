char mystring[] = "string"; // ок

////////

#include <iostream>
int main()
{
    char mystring[] = "string";
    std::cout << mystring << " has " << sizeof(mystring) << " characters.\n";
    for (int index = 0; index < sizeof(mystring); ++index)
        std::cout << static_cast<int>(mystring[index]) << " ";
    std::cout << std::endl;
    return 0;
}

///////

#include <iostream>
int main()
{
    char mystring[] = "string";
    mystring[1] = 'p';
    std::cout << mystring;
    return 0;
} // spring

///////

char name[255]; // объявляем достаточно большой массив (для хранения 255 символов)
std::cout << "Enter your name: ";
std::cin.getline(name, 255);
// будет принимать до 254 символов в массив name (оставляя место для нуль-терминатора!).

///////

// strcpy_s() позволяет копировать содержимое одной строки в другую, используется для присваивания значений строке:
#include <iostream>
#include <cstring>
int main()
{
    char text[] = "Print this!";
    char dest[50];
    strcpy_s(dest, text);
    std::cout << dest; // выводим "Print this!"
    return 0;
}

///////

// strlen() - возвращает длину строки C-style (без учета нуль-терминатора)
#include <iostream>
#include <cstring>
int main()
{
    char name[15] = "Max"; // используется только 4 символа (3 буквы + нультерминатор)
    std::cout << "My name is " << name << '\n';
    std::cout << name << " has " << strlen(name) << " letters.\n";
    std::cout << name << " has " << sizeof(name) << " characters in the array.\n";
    return 0;
}

///////

// strcat() — добавляет одну строку к другой (опасно);
// strncat() — добавляет одну строку к другой (с проверкой размера места назначения);
// strcmp() — сравнивает две строки (возвращает 0, если они равны);
// strncmp() — сравнивает две строки до определенного количества символов (возвращает 0, если до указанного символа не было различий)
// remove_prefix() — удаляет символы из левой части представления;
// remove_suffix() — удаляет символы из правой части представления.

//////

// копирование строки классом string_view
#include <iostream>
#include <string_view>
int main()
{
    std::string_view text{ "hello" }; // представление для строки "hello", которое хранится в бинарном виде
    std::string_view str{ text }; // представление этой же строки - "hello"
    std::string_view more{ str }; // представление этой же строки - "hello"
    std::cout << text << ' ' << str << ' ' << more << '\n';
    return 0;
}

//////

// Конвертация std::string_view в std::string
// void print_t(std::string s)
// {
    // std::cout << s << '\n';
// }
// std::string_view sv{ "balloon" };
// std::string str{ sv }; // явное преобразование
// print_t(static_cast<std::string>(sv)); // ок


///////


// Конвертация std::string_view в строку C-style
// std::string_view sv{ "balloon" };
// Создание объекта std::string из объекта std::string_view
// std::string str{ sv };
// Получаем строку C-style с нуль-терминатором
// auto szNullTerminated{ str.c_str() };

///////

