#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>

using namespace std;

int main()
{   
    string line;
    ifstream input("input.txt");

    // while (getline(input, line))
    // {
    //     cout << line << endl;
    // }

    ofstream output("output.txt");
    if (input)
    {   
        string line;
        while (getline(input, line))
        {   
            output << line << endl;
        }
    }
    return 0;
}