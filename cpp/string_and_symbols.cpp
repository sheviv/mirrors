#include <iostream>
#include <string>
#include <regex>
#include <cctype>
#include <bits/stdc++.h>

using namespace std;

int main()
{
    const int STRLEN = 200;
    char s[STRLEN];
    cin.getline(s, STRLEN);

    size_t maxlen = 0;
    char * maxidx = nullptr;

    for(char * c = s; *c;)
    {
        while(*c == ' ') ++c;
        if (*c == 0) break;
        char * begin = c;
        while(*c && *c != ' ') ++c;
        if (maxlen < (c - begin))
        {
            maxlen = c - begin;
            maxidx = begin;
        }
    }
    if (maxlen == 0)
    {
        // cout << "Empty line!\n";
        return 0;
    }
    else
    {
        *(maxidx+maxlen) = 0;
        cout << maxidx;
    }
}