#include <iostream>
#include <string>
#include <map>
#include <vector>

using namespace std;

void PrintBusesForStop(const map<string, vector<string>>& to_stops, map<string, vector<string>>& stop_bus, const string& stop)
{
    if (stop_bus.count(stop) == 0)
    {
        cout << "No stop" << endl;
    }
    else
    {
        for (const string& bus : stop_bus[stop])
        {
            cout << bus << " ";
        }
        cout << endl;
    }
}

void PrintStopsForBus(map<string, vector<string>>& to_stops, map<string, vector<string>>& stop_bus, const string& bus)
{
    if (to_stops.count(bus) == 0)
    {
        cout << "No bus" << endl;
    }
    else
    {
        for (const string& stop : to_stops[bus])
        {
            cout << "Stop " << stop << ": ";
            if (stop_bus[stop].size() == 1)
            {
                cout << "no interchange";
            }
            else
            {
                for (const string& other_bus : stop_bus[stop])
                {
                    if (bus != other_bus)
                    {
                        cout << other_bus << " ";
                    }
                }
            }
            cout << endl;
        }
    }
}
void PrintAllBuses(const map<string, vector<string>>& to_stops)
{
    if (to_stops.empty())
    {
        cout << "No buses" << endl;
    }
    else
    {
        for (const auto& item : to_stops)
        {
            cout << "Bus " << item.first << ": ";
            for (const string& stop : item.second)
            {
                cout << stop << " ";
            }
            cout << endl;
        }
    }
}

int main()
{
    int n;
    cin >> n;
    map<string, vector<string>> to_stops, stop_bus;
    for (int i = 0; i < n; ++i)
    {
        string bus_mode;
        cin >> bus_mode;
        if (bus_mode == "NEW_BUS")
        {
            string bus;
            cin >> bus;
            int c_stop;
            cin >> c_stop;
            vector<string>& stops = to_stops[bus];
            stops.resize(c_stop);
            for (string& stop : stops)
            {
                cin >> stop;
                stop_bus[stop].push_back(bus);
            }
        }
        else if (bus_mode == "BUSES_FOR_STOP")
        {
            string stop;
            cin >> stop;
            PrintBusesForStop(to_stops, stop_bus, stop);
        }
        else if (bus_mode == "STOPS_FOR_BUS")
        {
            string bus;
            cin >> bus;
            PrintStopsForBus(to_stops, stop_bus, bus);
        }
        else if (bus_mode == "ALL_BUSES")
        {
            PrintAllBuses(to_stops);
        }
    }
    return 0;
}