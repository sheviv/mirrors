#include <iostream>
#include <string>
#include <vector>

using namespace std;

const vector<int> month_l = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
const int month_s = month_l.size();

int main() {
    int q;
    cin >> q;
    // months index
    int month = 0;

    // внешний вектор должен иметь длину, равную количеству дней в первом месяце;
    // все внутренние векторы по умолчанию пусты, потому что дел изначально нет
    vector<vector<string>> days_do(month_l[month]);

    for (int i = 0; i < q; ++i)
    {
        string mode_of_day;
        cin >> mode_of_day;

        if (mode_of_day == "ADD")
        {
            int day;
            string concern;
            cin >> day >> concern;
            --day;
            days_do[day].push_back(concern);
        }
        else if (mode_of_day == "NEXT")
        {
            // перед переходом к следующему месяцу запомним длину предыдущего
            // обьявляем эту переменную константной, потому что менять её не планируем
            const int old_month_length = month_l[month];

            // номер месяца должен увеличиться на 1, но после декабря идёт январь:
            // например, (5 + 1) % 12 = 6, но (11 + 1) % 12 = 0
            month = (month + 1) % month_s;

            const int new_month_length = month_l[month];

            // если новый месяц больше предыдущего, достаточно сделать resize;
            // иначе перед resize надо переместить дела с «лишних» последних дней
            if (new_month_length < old_month_length)
            {
                // далее понадобится добавлять новые дела в последний день нового месяца
                // чтобы не писать несколько раз days_do[new_month_length - 1],
                // создадим ссылку с более коротким названием для этого вектора
                vector<string>& last_day_concerns = days_do[new_month_length - 1];

                // перебираем все «лишние» дни в конце месяца
                for (int day = new_month_length; day < old_month_length; ++day)
                {
                    // копируем вектор days_do[day]
                    // в конец вектора last_day_concerns
                    last_day_concerns.insert(end(last_day_concerns), begin(days_do[day]), end(days_do[day]));
                }
            }
            days_do.resize(new_month_length);
        } else if (mode_of_day == "DUMP")
        {
            int day;
            cin >> day;
            --day;
            // выводим список дел в конкретный день в нужном формате
            cout << days_do[day].size() << " ";
            for (const string& concern : days_do[day])
            {
                cout << concern << " ";
            }
            cout << endl;
        }
    }
    return 0;
}
