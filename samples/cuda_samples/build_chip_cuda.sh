# Please set the environment parameters for chip-spv installation and Intel runtime,
# i.e. CHIP_SPV_INSTALL and INTEL_RUNTIME
# export CHIP_SPV_INSTALL=/gpfs/jlse-fs0/users/ac.jzhao1/chip_workspace/chip-spv/install
# export INTEL_RUNTIME=/soft/libraries/intel-gpu-umd/f81b779-2022.06.09/driver/lib64/

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CHIP_SPV_INSTALL}/lib
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${INTEL_RUNTIME}
export CHIP_SPV_INCLUDE=${CHIP_SPV_INSTALL}/include

clang++ -I${CHIP_SPV_INCLUDE} -I${CHIP_SPV_INCLUDE}/hip -I${CHIP_SPV_INCLUDE}/cuspv -Wno-duplicate-decl-specifier -Wno-tautological-constant-compare  -Wno-c++20-extensions -Wno-unused-result -Wno-delete-abstract-non-virtual-dtor -Wno-deprecated-declarations -Wunused-command-line-argument -fPIE -Wno-format-extra-args -pthread -D__HIP_PLATFORM_SPIRV__= -x hip --target=x86_64-linux-gnu -Xclang -no-opaque-pointers --offload=spirv64 -nohipwrapperinc --hip-path=${CHIP_SPV_INSTALL} -std=c++17 -MD -MT $1.cu.o -MF $1.cu.o.d -o $1.cu.o -c $1.cu

clang++ -Wno-duplicate-decl-specifier -Wno-tautological-constant-compare  -Wno-c++20-extensions -Wno-unused-result -Wno-delete-abstract-non-virtual-dtor -Wno-deprecated-declarations -Wunused-command-line-argument --hip-path=${CHIP_SPV_INSTALL} $1.cu.o -o cu$1  -Wl,-rpath,${CHIP_SPV_INSTALL}: ${CHIP_SPV_INSTALL}/lib/libCHIP.so ${INTEL_RUNTIME}/libze_loader.so ${INTEL_RUNTIME}/libOpenCL.so -pthread
