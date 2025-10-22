// Stage 0 exercise: make both CPU and GPU print a greeting.
// Expected output (order matches host first, then 10 GPU threads):
// Stage 0: nvcc builds host and device code in one program.
// Hello from the CPU (host function)!
// Hello from the GPU (block 0, thread 0)!
// ...
// Hello from the GPU (block 0, thread 9)!

#include "todo.cu"

int main() {
    return main_todo();
}

