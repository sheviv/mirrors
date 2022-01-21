// #include <string>
// #include <iostream>
// #include <cmath> // для функции sqrt()
// int main()
// {
//     double z;
// tryAgain: // это лейбл
//     std::cout << "Enter a non-negative number: ";
//     std::cin >> z;
//     if (z < 0.0)
//         goto tryAgain; // а это оператор goto
//     std::cout << "The sqrt of " << z << " is " << sqrt(z) << std::endl;
//     return 0;
// }

////////

// #include <iostream>
// int main()
// {
//     int outer = 5;
//     while (outer >= 1)
//     {
//         int inner = 1;
//         while (inner <= outer)
//             std::cout << inner++ << " ";
//         std::cout << "\n";
//         --outer;
//     }
//     return 0;
// }
// 1 2 3 4 5 
// 1 2 3 4 
// 1 2 3 
// 1 2 
// 1

///////

// #include <iostream>
// int main()
// {
    // int outer = 5;
    // while (outer >= 1)
    // {
        // int inner = outer;
        // while (inner >= 1)
            // std::cout << inner-- << " ";
        // std::cout << "\n";
        // --outer;
    // }
    // return 0;
// }
// 5 4 3 2 1 
// 4 3 2 1 
// 3 2 1 
// 2 1 
// 1 

////////


// #include <iostream>
// int main()
// {
//     int outer = 1;
//     while (outer <= 5)
//     {
//         int inner = 5;
//         while (inner >= 1)
//         {
//             // Первое число в любом ряде совпадает с номером этого ряда, поэтому числа должны выводиться только если <= номера ряда (в противном случае, выводится пробел)
//             if (inner <= outer)
//                 std::cout << inner << " ";
//             else
//                 std::cout << " "; // вставляем дополнительные пробелы
//             --inner;
//         }
//         std::cout << "\n";
//         ++outer;
//     }
// }
//         1 
//       2 1 
//     3 2 1 
//   4 3 2 1 
// 5 4 3 2 1

/////////

