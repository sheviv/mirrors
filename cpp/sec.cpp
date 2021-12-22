#include <iostream>
#include <stdio.h> 
#include <cmath>
#include <iomanip>
#include <limits>
#include <vector>
#include <algorithm> 

using namespace std;

int main()
{
	int n, m, ii, jj;
    vector <int> v;
	cin >> n;
    cin >> m;
	int a[n][m];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            cin >> a[i][j];
        }
    }
    cin >> ii;
    cin >> jj;

    for (int j = 0; j < a[i].size(); j++)
    {
        if (trans_vec[j].size() != b.size())
            trans_vec[j].resize(b.size());
        trans_vec[j][i] = b[i][j];
    }
	return 0;
}