// #include <Python.h>
#include "Python.h"
// #include "numpy/arrayobject.h"

#include <iostream>
#include <string>
 
static PyObject * 
printString(PyObject * self, PyObject* args)
{
    std::cout << PyString_AsString(args) << std::endl;
    return Py_BuildValue("");
}

// static PyMethodDef EmbMethods[] = {
//     {"printMessage", printString},
//     {NULL, NULL}
// };
void PyMethodDef()
{
    std::cout << "PyMethodDef" << std::endl;
}


// int main()
// {
//     std::cout << "asd" << std::endl;
//     return 0;
// }