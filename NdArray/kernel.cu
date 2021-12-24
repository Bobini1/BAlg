
#include "Algorithms.cuh"
#include <iostream>
#include <vector>

int main()
{
    auto* arr = new double[10000000];
    for (int i = 0; i < 10000000; i++)
    {
        arr[i] = i + 1;
    }
    
    std::cout << BAlg::Algorithms::reduce(arr, 1, BAlg::Algorithms::Operation::ADD) << std::endl;

    return 0;
}