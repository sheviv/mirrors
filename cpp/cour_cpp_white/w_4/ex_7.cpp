#include <iostream>
#include <string>
#include <vector>

using namespace std;


// struct Student {Student(string first_name, string second_name, int bd_day, int bd_month, int bd_year)
//     {
//     name = first_name;
//     s_name = second_name;
//     day = bd_day;
//     month = bd_month;
//     year = bd_year;
//     }
//     string name = "";
//     string s_name = "";
//     int day = 0;
//     int month = 0;
//     int year = 0;
// };

struct Student
{
    string name;
    string s_name;

    int day;
    int month;
    int year;
};

// int main()
// {
//     string name, s_name;
//     int day, month, year;
//     vector<Student> v;

//     int n;
//     cin >> n;
//     for (int i = 0; i < n; ++i)
//     {
//         cin >> name >> s_name >> day >> month >> year;
//         Student student{name, s_name, day, month, year};
//         v.push_back(student);
//     }

//     int m;
//     cin >> m;
//     for (int j = 0; j < m; ++j)
//     {
//         string mode;
//         int k;
//         cin >> mode >> k;
//         if (mode == "name" && k > 0 && k <= v.size())
//         {
//             cout << v[k-1].name << " " << v[k-1].s_name << endl;
//         }
//         else if (mode == "date" && k > 0 && k <= v.size())
//         {
//             cout << v[k-1].day << "." << v[k-1].month << "." << v[k-1].year << endl;
//         }
//         else
//         {
//             cout << "bad request" << endl;
//         }
//     }
//     return 0;
// }

int main() {
  int n;
  cin >> n;

  string first_name, last_name;
  int day, month, year;
  vector<Student> students;

  for (size_t i = 0; i < n; ++i) {
    cin >> first_name >> last_name >> day >> month >> year;

    students.push_back(Student{
      first_name,
      last_name,
      day,
      month,
      year
    });
  }


  int m;
  cin >> m;
  string query;
  int student_number;

  for (int i = 0; i < m; ++i) {
    cin >> query >> student_number;
    --student_number;

    if (query == "name" && student_number >= 0 && student_number < n) {
      cout << students[student_number].first_name << " "
           << students[student_number].last_name << endl;
    } else if (query == "date" && student_number >= 0 && student_number < n) {
      cout << students[student_number].day << "."
           << students[student_number].month << "."
           << students[student_number].year << endl;
    } else {
      cout << "bad request" << endl;
    }
  }

  return 0;
}