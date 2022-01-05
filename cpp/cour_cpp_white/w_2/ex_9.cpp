#include <iostream>
#include <string>
#include <vector>

using namespace std;

const vector<int> month_l = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
const int month_s = month_l.size();

int main() {
    int q;
    cin >> q;
    int month = 0;
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
            const int old_month_length = month_l[month];
            month = (month + 1) % month_s;

            const int new_month_length = month_l[month];
            if (new_month_length < old_month_length)
            {
                vector<string>& last_day_concerns = days_do[new_month_length - 1];
                for (int day = new_month_length; day < old_month_length; ++day)
                {
                    last_day_concerns.insert(end(last_day_concerns), begin(days_do[day]), end(days_do[day]));
                }
            }
            days_do.resize(new_month_length);
        } else if (mode_of_day == "DUMP")
        {
            int day;
            cin >> day;
            --day;
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
