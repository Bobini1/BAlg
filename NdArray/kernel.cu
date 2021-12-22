
#include "Algorithms.cuh"
#include <iostream>

int main()
{
    long* arr = new long[1000000];
    long sum = 0;
    for (int i = 0; i < 1000000; i++)
    {
        arr[i] = i + 1;
        sum += i + 1;
    }
    
    std::cout << sum << " " << BAlg::Algorithms::reduce(arr, 1000000, BAlg::Algorithms::Operation::ADD) << std::endl;

    return 0;
}