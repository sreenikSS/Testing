#include <iostream>
#include <string>
#include <iterator>
#include <map>
#include <boost/optional.hpp>
using namespace std;
void initializeMap(map<string, double>& params)
{
  params["var1"] = 5;
  params["var2"] = -2;
  params["var3"] = 0;
  params["var4"]; // I want to explicitly retain the information that this is not initialized
  params["var5"] = NAN;// Seems like the method in the previous line doesn't work so this can be an alternative
}
int main()
{
  map<string, double> params;
  initializeMap(params); // Map initialized with the variables

  map<string, double>::iterator itr;
  for (itr = params.begin(); itr != params.end(); ++itr)
  {
    cout << "\nKey: " << itr->first << " Value: " << itr->second << "\n";
    if (itr->second == NULL)
    {
      cout << "Key " << itr->first << " identified as having a value of NULL\n";
    }
    if (itr->second == 0)
    {
      cout << "Key " << itr->first << " identified as having a value 0\n";
    }
  }
  // The output is as follows:
  /*

  Key: var1 Value: 5

  Key: var2 Value: -2

  Key: var3 Value: 0
  Key var3 identified as having a value of NULL
  Key var3 identified as having a value 0

  Key: var4 Value: 0
  Key var4 identified as having a value of NULL
  Key var4 identified as having a value 0

  Key: var5 Value: nan
  */

  //Seems like assigning to nan helps me differentiate between an assigned value and an yet-unassigned value
  //Another technique however will be to use boost/optional
  return 0;
}
