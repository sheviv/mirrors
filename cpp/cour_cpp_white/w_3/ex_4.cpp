#include <map>
#include <string>

using namespace std;

string FindNameByYear(const map<int, string>& names, int year) {
    string name;  // изначально имя неизвестно
    // перебираем всю историю по возрастанию ключа словаря, то есть в хронологическом порядке
    for (const auto& item : names)
    {
        // если очередной год не больше данного, обновляем имя
        if (item.first <= year)
        {
            name = item.second;
        }
        else
        {
            // иначе пора остановиться, так как эта запись и все последующие относятся к будущему
            break;
        }
    }
    return name;
}

class Person {
public:
    void ChangeFirstName(int year, const string& f_name)
    {
        f_names[year] = f_name;
    }
    void ChangeLastName(int year, const string& l_name)
    {
        l_names[year] = l_name;
    }
    string GetFullName(int year)
    {
        // получаем имя и фамилию по состоянию на год year
        const string f_name = FindNameByYear(f_names, year);
        const string l_name = FindNameByYear(l_names, year);
        // если и имя, и фамилия неизвестны
        if (f_name.empty() && l_name.empty())
        {
            return "Incognito";
            // если неизвестно только имя
        }
        else if (f_name.empty())
        {
            return l_name + " with unknown first name";
            // если неизвестна только фамилия
        }
        else if (l_name.empty())
        {
            return f_name + " with unknown last name";
            // если известны и имя, и фамилия
        }
        else
        {
            return f_name + " " + l_name;
        }
    }

private:
    map<int, string> f_names;
    map<int, string> l_names;
};