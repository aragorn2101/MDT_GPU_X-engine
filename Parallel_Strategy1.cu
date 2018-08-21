#include <cuda.h>
#include <cufft.h>


/*
 *  X-engine kernel -- Parallel Strategy 1
 *
 *  Nchannels: number of channels in 1 spectrum,
 *  Nspectra: total number of spectra within 1 integration time,
 *  XInput: input array for X-tengine coming directly from F-engine,
 *  FXOutput: output array for the whole FX operation.
 *
 *  The thread grid is setup using the following:
 *
 *  if (Nchannels > MaxThreadsPerBlock)
 *  {
 *    NumBlockz = Nchannels / MaxThreadsPerBlock + 1;
 *    NumThreadx = Nchannels / NumBlockz;
 *  }
 *  else
 *  {
 *    NumBlockz = 1;
 *    NumThreadx = Nchannels;
 *  }
 *  NumBlockx  = Nelements;
 *  NumBlocky  = Nelements;
 *  NumThready = 1;
 *  NumThreadz = 1;
 *
 *  And it is lauched with the following call:
 *  XEngine<<< dim3(NumBlockx,NumBlocky,NumBlockz), dim3(NumThreadx,NumThready,NumThreadz), ShMemPerBlock >>>(Nchannels, Nspectra, d_XInput, d_FXOutput);
 *
 */
__global__ void XEngine(int Nchannels, int Nspectra, cufftComplex *XInput, float2 *FXOutput)
{
  int i = blockIdx.z*blockDim.x + threadIdx.x;
  int out_stride;
  cufftComplex tmp_input1, tmp_input2;
  float2 tmp_output;

  if (blockIdx.x >= blockIdx.y && i < Nchannels)
  {
    tmp_output.x = 0.0f;
    tmp_output.y = 0.0f;

    for ( i=0 ; i<Nspectra ; i++ )
    {
      /*  Read input for element i (blockIdx.x)  */
      tmp_input1 = XInput[(blockIdx.y*Nspectra + i)*Nchannels + blockIdx.z*blockDim.x + threadIdx.x];
      tmp_input2 = XInput[(blockIdx.x*Nspectra + i)*Nchannels + blockIdx.z*blockDim.x + threadIdx.x];

      /*  Cross multiplication and accumulation  */
      /*  + tmp_input1 x conj(tmp_input2)  */
      /*  Re  */
      tmp_output.x = fmaf(tmp_input1.x, tmp_input2.x, tmp_output.x);
      tmp_output.x = fmaf(tmp_input1.y, tmp_input2.y, tmp_output.x);

      /*  Im  */
      tmp_output.y = fmaf(tmp_input1.y,  tmp_input2.x, tmp_output.y);
      tmp_output.y = fmaf(tmp_input1.x, -tmp_input2.y, tmp_output.y);
    }

    tmp_output.x /= Nspectra;
    if (blockIdx.x != blockIdx.y)
      tmp_output.y /= Nspectra;

    out_stride = (int)( ((0.5f*blockIdx.y * (2.0f*gridDim.x - blockIdx.y + 1.0f)) + blockIdx.x - blockIdx.y) * Nchannels );

    /*  Write output to global memory  */
    FXOutput[out_stride + threadIdx.x] = tmp_output;
  }
}
