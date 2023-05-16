#include <iostream>

class SmallAllocator {
private:
        char Memory[1048576];
        unsigned int start_of_free = 0;
public:
        void *Alloc(unsigned int Size) {
        	unsigned int current_start = start_of_free;
        	start_of_free += Size;
        	if (start_of_free >= 1048576)
        	 {
        	 	start_of_free = current_start;
        	 	return nullptr;
        	 } 
        	return (void*) ((char*)Memory + current_start);
        };
        void *ReAlloc(void *Pointer, unsigned int Size) {
        	if (start_of_free + Size >= 1048576)
        	{
        		return nullptr;
        	}
        	for (uint i = 0; i < Size; i++)
        	{
        		*(Memory + start_of_free + i) = *( (char*)Pointer + i);
        	}
        	start_of_free += Size;
        	return (void*) ((char*)Memory + start_of_free - Size) ;
        };
        void Free(void *Pointer) {};
}; 

int main(int argc, char const *argv[])
{
	SmallAllocator A1;
	int * A1_P1 = (int *) A1.Alloc(sizeof(int));
	A1_P1 = (int *) A1.ReAlloc(A1_P1, 2 * sizeof(int));
	A1.Free(A1_P1);
	SmallAllocator A2;
	int * A2_P1 = (int *) A2.Alloc(10 * sizeof(int));
	for(unsigned int i = 0; i < 10; i++) A2_P1[i] = i;
	for(unsigned int i = 0; i < 10; i++) if(A2_P1[i] != i) std::cout << "ERROR 1" << std::endl;
	int * A2_P2 = (int *) A2.Alloc(10 * sizeof(int));
	for(unsigned int i = 0; i < 10; i++) A2_P2[i] = -1;
	for(unsigned int i = 0; i < 10; i++) if(A2_P1[i] != i) std::cout << "ERROR 2" << std::endl;
	for(unsigned int i = 0; i < 10; i++) if(A2_P2[i] != -1) std::cout << "ERROR 3" << std::endl;
	A2_P1 = (int *) A2.ReAlloc(A2_P1, 20 * sizeof(int));
	for(unsigned int i = 10; i < 20; i++) A2_P1[i] = i;
	for(unsigned int i = 0; i < 20; i++) if(A2_P1[i] != i) std::cout << "ERROR 4" << std::endl;
	for(unsigned int i = 0; i < 10; i++) if(A2_P2[i] != -1) std::cout << "ERROR 5" << std::endl;
	A2_P1 = (int *) A2.ReAlloc(A2_P1, 5 * sizeof(int));
	for(unsigned int i = 0; i < 5; i++) if(A2_P1[i] != i) std::cout << "ERROR 6" << std::endl;
	for(unsigned int i = 0; i < 10; i++) if(A2_P2[i] != -1) std::cout << "ERROR 7" << std::endl;
	A2.Free(A2_P1);
	A2.Free(A2_P2);
	return 0;
}