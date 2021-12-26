
#include "Algorithms.cuh"
#include "NdArray.cuh"
#include <iostream>

int main()
{
    /**auto* arr = new double[1000];
    for (int i = 0; i < 1000; i++)
    {
        arr[i] = i + 1;
    }**/

    BAlg::DataStructures::NdArray<std::string, 3> test({2, 4, 6});
    test[0][1][2] = "zzz";
    BAlg::DataStructures::NdArray<std::string, 2> test2 = test[0];
    BAlg::DataStructures::NdArray<std::string, 1> test3 = test[0][1];
    test3[2] = "aaa";
    std::string x = test[0][1][2];

    std::cout << x << std::endl << test3[2];
    
    //std::cout << BAlg::Algorithms::reduce(arr, 100, BAlg::Algorithms::Operation::ADD) << std::endl;

    return 0;
}