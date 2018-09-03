/*
 *  CUDA kernel for X-engine using parallel strategy 1.
 *
 *  Copyright (C) 2018 Nitish Ragoomundun
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU Affero General Public License as
 *  published by the Free Software Foundation, either version 3 of the
 *  License, or (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Affero General Public License for more details.
 *
 *  You should have received a copy of the GNU Affero General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 */

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
