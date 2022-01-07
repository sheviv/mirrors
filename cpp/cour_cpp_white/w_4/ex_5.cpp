#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <iomanip>

using namespace std;

int main()
{   
    string line;
    ifstream input("input.txt");
    vector<double> values;

    while (getline(input, line))
    {   
        double i = std::stod(line);
        values.push_back(i);
    }

    cout << fixed << setprecision(3);
    for (auto i : values)
    {
        cout << i << endl;
    }

    // ofstream output("output.txt");s
    // if (input)
    // {   
    //     string line;
    //     while (getline(input, line))
    //     {   
    //         output << line << endl;
    //     }
    // }
    return 0;
}