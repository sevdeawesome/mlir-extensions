# RFC: GPUX Dialect (Extension of upstream GPU Dialect)

Core MLIR Team

## Summary

We propose the GPUX dialect as an extension of the upstream GPU dialect. The GPUX dialect allows us to expose stream creation/destruction ops.These ops are required by the underlying runtimes (Level zero & Sycl) for explicit stream/context/device creation.
The GPUX dialect will also have some of the upstream GPU dialect ops extended with an added argument for stream in those ops.

## Motivation

Upstream MLIR has a Dialect called the [GPU Dialect](https://mlir.llvm.org/docs/Dialects/GPU/) which provides middle-level abstractions for launching GPU kernels following a programming model similar to that of CUDA or OpenCL. The upstream dialect exposes all the operation required to launch the kernel on a GPU device. But, the drawback with the upstream dialect is it does not expose Stream as an operation in the dialect. For example, if the user wants to launch the kernel on a particular stream(provided by the user), there is no way to do that today. For the upstream dialect, stream is created internally while lowering to LLVM runtime calls and it refers specifically to the CUDA stream. Therefore, the need to create a GPUX Dialect arises. The GPUX dialect is an extension of the upstream GPU dialect with following ops - CreateStreamOp, DestroyStreamOp, GPUXLaunchFuncOp, GPUXAllocOp & GPUXDeallocOp . The CreateStreamOp op allows for explicit Stream creation. The stream is a default stream if its not provided by user or it can be a user provided stream. The stream here is basically a SYCL/L0 queue, with default context and device. If the stream is user provided we assume the user is adhering to the Stream Type (!gpux.StreamType which is a OpaqueType pointer).

## Proposal

We propose to have the GPUX dialect with ops listed below:

### Operation:

#### gpux.create_stream (gpux::CreateStreamOp)

Create the GPU Stream.

Syntax:

```
operation ::= "gpux.create_stream"() : () -> !gpux.StreamType
```

The gpux.create_stream() operation will be added by a custom pass only if the stream is not provided by the user.
If user does not provide the stream, this operation will be converted to runtime call for default stream creation in the LLVM generated code. The runtime routine will create a SYCL/L0 queue, device and context based on the runtime.


#### gpux.destroy_stream (gpux::DestroyStreamOp)

Destroy the GPU Stream.

Syntax:

```
operation ::= "gpux.destroy_stream"() : () -> !gpux.StreamType
```

The gpux.destroy_stream() operation will be added by a custom pass only if the stream is not provided by the user. If the user provides the stream, we expect the user to manage the lifetime of the stream.
If user does not provide the stream, this operation will destroy the default stream.


#### gpux.alloc (gpux::AllocOp)

GPU memory allocation operation.

Syntax:

```
operation ::= gpux.alloc custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
              ($gpux_stream, $dynamicSizes) ([ $symbolOperands^ ])? attr-dict : type($memref)
```

This op is an extension of upstream gpu.alloc op with one added argument for stream. Stream here will be an optional argument, so when upstreaming the code, the upstream GPU dialect users have no change in their usage of this op.


#### gpux.dealloc (gpux::DeallocOp)

GPU memory deallocation operation.

Syntax:

```
operation ::= gpux.dealloc custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
              ($gpux_stream, $memref) attr-dict : type($memref)
```

This op is an extension of upstream gpu.dealloc op with one added argument for stream. Stream here will be an optional argument, so when upstreaming the code, the upstream GPU dialect users have no change in their usage of this op.


#### gpux.launch_func (gpux::LaunchFuncOp)

Launches a function as a GPU kernel.

Syntax:

```
operation ::= gpux.launch_func custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
              $kernel
              blocks in  ($gridSizeX, $gridSizeY, $gridSizeZ)
              threads in ($blockSizeX, $blockSizeY, $blockSizeZ)
              (dynamic_shared_memory_size $dynamicSharedMemorySize^)?
              custom<LaunchFuncOperands>($gpux_stream, $operands, type($operands)) attr-dict
```

This op is an extension of upstream gpu.launch_func op with one added argument for stream. Stream here will be an optional argument, so when upstreaming the code, the upstream GPU dialect users have no change in their usage of this op.

#### gpux.wait (gpux::WaitOp)

Wait for gpu ops in a particular stream to complete.

Syntax:

```
operation ::= gpu.wait custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
              $gpux_stream attr-dict
```

The wait op will need to be provided with the stream to wait on. This op is an extension of upstream gpu.wait op with one added argument for stream. Stream here will be an optional argument, so when upstreaming the code, the upstream GPU dialect users have no change in their usage of this op.

## Example Usage in case of user provided stream:

// IR before our custom transformation pass:

```

func.func @main(%stream : !gpux.StreamType) attributes {llvm.emit_c_interface} {
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 2.200000e+00 : f32
  %cst_0 = arith.constant 1.100000e+00 : f32
  %cst_1 = arith.constant 0.000000e+00 : f32
  %0 = gpu.alloc  () : memref<8xf32>
  %1 = gpu.alloc  () : memref<8xf32>
  %2 = gpu.alloc  () : memref<8xf32>
  call @fillResource1DFloat(%0, %cst_0) : (memref<?xf32>, f32) -> ()
  call @fillResource1DFloat(%1, %cst) : (memref<?xf32>, f32) -> ()
  call @fillResource1DFloat(%2, %cst_1) : (memref<?xf32>, f32) -> ()
  gpu.launch_func  @main_kernel::@main_kernel blocks in (%c8, %c1, %c1) threads in (%c1, %c1, %c1) args(%0 : memref<8xf32>, %1 : memref<8xf32>, %2 : memref<8xf32>)
  %3 = memref.cast %2 : memref<8xf32> to memref<*xf32>
  call @printMemrefF32(%3) : (memref<*xf32>) -> ()
  gpu.dealloc(%0) : memref<8xf32>
  gpu.dealloc(%1) : memref<8xf32>
  gpu.dealloc(%2) : memref<8xf32>
  return
}

```

// IR after our custom transformation pass:

```

func.func @main(%stream : !gpux.StreamType) attributes {llvm.emit_c_interface} {
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 2.200000e+00 : f32
  %cst_0 = arith.constant 1.100000e+00 : f32
  %cst_1 = arith.constant 0.000000e+00 : f32
  %0 = gpux.alloc (%stream) : (!gpux.StreamType) -> memref<8xf32>
  %1 = gpux.alloc (%stream) : (!gpux.StreamType) -> memref<8xf32>
  %2 = gpux.alloc  (%stream) : (!gpux.StreamType) -> memref<8xf32>
  call @fillResource1DFloat(%0, %cst_0) : (memref<?xf32>, f32) -> ()
  call @fillResource1DFloat(%1, %cst) : (memref<?xf32>, f32) -> ()
  call @fillResource1DFloat(%2, %cst_1) : (memref<?xf32>, f32) -> ()
  gpux.launch_func  @main_kernel::@main_kernel blocks in (%c8, %c1, %c1) threads in (%c1, %c1, %c1) args(%stream : !gpux.StreamType, %memref : memref<8xf32>, %memref_2 : memref<8xf32>, %memref_3 : memref<8xf32>)
  %3 = memref.cast %memref_3 : memref<8xf32> to memref<*xf32>
  call @printMemrefF32(%3) : (memref<*xf32>) -> ()
  gpux.dealloc(%stream, %0) : (!gpux.StreamType, memref<8xf32>) -> ()
  gpux.dealloc(%stream, %1) : (!gpux.StreamType, memref<8xf32>) -> ()
  gpux.dealloc(%stream, %2) : (!gpux.StreamType, memref<8xf32>) -> ()
  return
}
```

Since the stream is already provided by the user in this case, only the alloc, dealloc and the launch_func operations were translated to the extended versions in the GPUX dialect.


## Example Usage in case of default stream:

// IR before our custom transformation pass:

```

func.func @main() attributes {llvm.emit_c_interface} {
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 2.200000e+00 : f32
  %cst_0 = arith.constant 1.100000e+00 : f32
  %cst_1 = arith.constant 0.000000e+00 : f32
  %0 = gpu.alloc  () : memref<8xf32>
  %1 = gpu.alloc  () : memref<8xf32>
  %2 = gpu.alloc  () : memref<8xf32>
  gpu.launch_func  @main_kernel::@main_kernel blocks in (%c8, %c1, %c1) threads in (%c1, %c1, %c1) args(%0 : memref<8xf32>, %1 : memref<8xf32>, %2 : memref<8xf32>)
  %3 = memref.cast %2 : memref<8xf32> to memref<*xf32>
  call @printMemrefF32(%3) : (memref<*xf32>) -> ()
  gpu.dealloc(%0) : memref<8xf32>
  gpu.dealloc(%1) : memref<8xf32>
  gpu.dealloc(%2) : memref<8xf32>
  return
}

```

// IR after our custom transformation pass:

```

func.func @main() attributes {llvm.emit_c_interface} {
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 2.200000e+00 : f32
  %cst_0 = arith.constant 1.100000e+00 : f32
  %cst_1 = arith.constant 0.000000e+00 : f32
  %stream = gpux.create_stream () : () -> !gpux.StreamType
  %0 = gpux.alloc (%stream) : (!gpux.StreamType) -> memref<8xf32>
  %1 = gpux.alloc (%stream) : (!gpux.StreamType) -> memref<8xf32>
  %2 = gpux.alloc  (%stream) : (!gpux.StreamType) -> memref<8xf32>
  gpux.launch_func  @main_kernel::@main_kernel blocks in (%c8, %c1, %c1) threads in (%c1, %c1, %c1) args(%stream : !gpux.StreamType, %memref : memref<8xf32>, %memref_2 : memref<8xf32>, %memref_3 : memref<8xf32>)
  %3 = memref.cast %memref_3 : memref<8xf32> to memref<*xf32>
  call @printMemrefF32(%3) : (memref<*xf32>) -> ()
  gpux.dealloc(%stream, %0) : (!gpux.StreamType, memref<8xf32>) -> ()
  gpux.dealloc(%stream, %1) : (!gpux.StreamType, memref<8xf32>) -> ()
  gpux.dealloc(%stream, %2) : (!gpux.StreamType, memref<8xf32>) -> ()
  gpux.stream.destroy(%stream) : (!gpux.StreamType) -> ()
  return
}
```


## Features to be added in subsequent PR's (either in the dialect or via a pass operating on this dialect):
1. More ops can be added/extended to dialect going forward based on the use cases and requirements.
2. Currently, the stream creation and destruction happens for every function. A pass to manage stream will be added where the stream is created only once at the beginning of the program and used throughout.

## Upstreaming Plans.

We intend to create an RFC for changing the upstream GPU dialect ops to include stream as an optional argument in the futute. Till then, we plan to use this extended dialect.

## Questions

Do we need more operations like context & devices? What if the user wants to pass context and device as well?
