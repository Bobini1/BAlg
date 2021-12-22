#include <vector>
#include <memory>

namespace BAlg {
	namespace DataStructures
	{
		template<typename T>
		class NdArray
		{
		public:
			NdArray(std::initializer_list<size_t> shape) :shape(shape)
			{
				size_t size;
				data = std::make_shared<T[]>(new T[size]);
			}
			NdArray(const NdArray& otherArray);
			NdArray& operator=(const NdArray& otherArray);
			NdArray(NdArray&& otherArray);
			NdArray& operator=(NdArray&& otherArray);
		private:
			std::vector<size_t> shape;
			std::shared_ptr<T[]> data;
		};
	}
}