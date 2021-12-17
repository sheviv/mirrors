#include <iostream>

using namespace std;

int main()
{
    int e, prev, curr_rep_len, max_rep_len, element;
    prev = -1;
    curr_rep_len = 0;
    max_rep_len = 0;
    cin >> e;
    element = e;
    while (element != 0)
    {
        if (prev == element)
        {
            curr_rep_len += 1;
        }
        else
        {
            prev = element;
            max_rep_len = max(max_rep_len, curr_rep_len);
            curr_rep_len = 1;
        }
        cin >> element;
    }
    max_rep_len = max(max_rep_len, curr_rep_len);
    cout << max_rep_len << endl;
}