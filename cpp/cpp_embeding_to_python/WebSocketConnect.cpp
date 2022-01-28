#include <Python.h>
#include <boost/python.hpp>
  
#include <iostream>
#include <string>
#include "PrintEmb.cpp" 
  
void WebSocketConnect()
{
    using namespace boost::python;
     
    Py_Initialize();
     
    // Py_InitModule("cppMethods", EmbMethods);
     
    PyObject * ws = PyImport_ImportModule("echo_client");
    std::string address = "ws://html5labs-interop.cloudapp.net:4502/chat";
    call_method<void>(ws, "Connect", address);
     
    Py_Finalize();
}