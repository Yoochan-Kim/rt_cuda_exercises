// Stage 1 exercise: allocate device memory, run a kernel, copy results back.

#include <iostream>
#include <vector>

#include "todo.cu"

int main() {
    constexpr int elementCount = 128;
    std::vector<int> hostData(elementCount, -1);

    if (!runWriteThreadIdsTest(elementCount, hostData)) {
        return 1;
    }

    bool passed = true;
    for (int i = 0; i < elementCount; ++i) {
        if (hostData[i] != i) {
            passed = false;
            break;
        }
    }

    if (passed) {
        std::cout << "Stage 1 thread ID write test passed ✅" << std::endl;
    } else {
        std::cout << "Stage 1 thread ID write test failed ❌" << std::endl;
    }

    return 0;
}
