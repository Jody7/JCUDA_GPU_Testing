import static jcuda.driver.CUdevice_attribute.*;
import static jcuda.driver.JCudaDriver.*;

import java.util.*;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaMemcpyKind;

public class GPUMain
{
    public static CUdeviceptr[] allocatePointers(int threadNum, int size){
        CUdeviceptr HDP[] = new CUdeviceptr[threadNum];
        for(int i = 0; i < threadNum; i++)
        {
            HDP[i] = new CUdeviceptr();
            JCuda.cudaMalloc(HDP[i], size * 4);
        }
        return HDP;
    }


    public static void main(String args[]) {
        JCudaDriver.setExceptionsEnabled(true);
        cuInit(0);

        int threads = 8;
        int[] hostData = new int[]{1};
        int size = hostData.length;

        //Display Device Name
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        byte dName[] = new byte[256];
        cuDeviceGetName(dName, 256, device);

        System.out.println(new String(dName));

        //allocate memory in the device
        CUdeviceptr HDP[] = allocatePointers(threads, size);


        //write data into device
        for(int i = 0; i < threads; i++)
        {
            cuMemcpyHtoD(HDP[i], CUdeviceptr.to(hostData), 4*size);
        }

        //prepare device buffer
        CUdeviceptr deviceInput = new CUdeviceptr();
        cuMemAlloc(deviceInput, threads * Sizeof.POINTER);
        cuMemcpyHtoD(deviceInput, Pointer.to(HDP),
                threads * Sizeof.POINTER);

        //prepare output memory in the device
        CUdeviceptr deviceOutput = new CUdeviceptr();
        cuMemAlloc(deviceOutput, threads * 4);

        CUmodule module = new CUmodule();
        cuModuleLoad(module, "C:\\addkern.ptx");

        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, module, "add");


        Pointer kernelParameters = Pointer.to(
                Pointer.to(deviceInput),
                Pointer.to(new int[]{size}),
                Pointer.to(deviceOutput)
        );


         cuLaunchKernel(function, 1, 1, 1, threads, 1, 1, 0, null, kernelParameters, null);

                                //Copy the device result into the host device
         int[] res = new int[threads];
         JCuda.cudaMemcpy(Pointer.to(res), deviceOutput, 4*threads*size, cudaMemcpyKind.cudaMemcpyDeviceToHost);

        for(int i=0; i<res.length; i++) {
            System.out.println(res[i]);
        }

    }
}
