    #include <iostream>
    #include <vector>
    #include <string>
    #include <algorithm>
    #include <cmath>


    using namespace std;

    int  main()
    {
        int a;
        int b;
        cin >> a >> b;
        if (b == 0)
        {
            cout << "Impossible" << endl;
        }
        else
        {
            cout << a / b << endl;
        }
        return 0;
    }