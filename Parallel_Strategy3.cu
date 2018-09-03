/*
 *  CUDA kernel for X-engine using parallel strategy 3.
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
 *  X-engine kernel -- Parallel Strategy 3
 *
 *  Making use of 4x4 tiles across the correlation matrix.
 *
 *  Nelements: number of elements in interferometer,
 *  Nchannels: number of channels in 1 spectrum,
 *  Nspectra: total number of spectra within 1 integration time,
 *  XInput: input array for X-tengine coming directly from F-engine,
 *  FXOutput: output array for the whole FX operation.
 *
 *  The thread grid is setup using the following:
 *
 *  NumThready = 4;
 *  NumThreadz = 4;
 *
 *  NumThreadx = MaxThreadsPerBlock / (NumThready*NumThreadz);  // 64 x 4 x  4
 *
 *  NumBlockx = (Nchannels/NumThreadx) + ((Nchannels%NumThreadx == 0) ? 0 : 1);
 *  NumBlocky = ((Nelements+3)/4) * ((Nelements+3)/4 + 1) / 2;
 *  NumBlockz = 1;
 *
 *  And it is lauched with the following call:
 *  XEngine<<< dim3(NumBlockx,NumBlocky,NumBlockz), dim3(NumThreadx,NumThready,NumThreadz) >>>(Nelements, Nchannels, Nspectra, d_XInput, d_FXOutput);
 *
 */
__global__ void XEngine(int Nelements, int Nchannels, int Nspectra, cufftComplex *XInput, float2 *FXOutput)
{
  __shared__ int TileIDx, TileIDy;

  /*  Shared memory for input data  */
  __shared__ cufftComplex sh_Input[512];

  int channelIdx = blockIdx.x*blockDim.x + threadIdx.x;
  int sharedIdx = threadIdx.y*256 + threadIdx.z*64 + threadIdx.x;
  int Elementi, Elementj;

  float2 tmp_input1, tmp_input2, tmp_output;

  long k;


  /***  BEGIN Initialize tile indices and output array  ***/
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
  {
    Elementj = blockIdx.y;  // blockID in x direction of matrix
    Elementi = 0;  // blockID in y direction of matrix
    k = (Nelements + 3)/4;
    while (Elementj >= k)
    {
      Elementj -= k;
      k--;
      Elementi++;
    }
    TileIDx = Elementj + Elementi;
    TileIDy = Elementi;
  }

  tmp_output.x = 0.0f;
  tmp_output.y = 0.0f;

  /***  END Initialize tile indices and output array  ***/


  __syncthreads();


  Elementi = TileIDy*4 + threadIdx.y;
  Elementj = TileIDx*4 + threadIdx.z;


  /***  BEGIN Work for threads within Nchannels  ***/
  if (channelIdx < Nchannels)
  {
    /***  BEGIN Loop through spectra, correlate and accumulate  ***/
    for ( k=0 ; k<Nspectra ; k++ )
    {
      /***  BEGIN Copy input data from global memory to shared memory  ***/
      if (TileIDx == TileIDy)
      {
        if (threadIdx.y == 0)
        {
          if (Elementi + threadIdx.z < Nelements)
            sh_Input[sharedIdx] = XInput[((Elementi + threadIdx.z)*Nspectra + k)*Nchannels + channelIdx];
          else
          {
            sh_Input[sharedIdx].x = 0.0f;
            sh_Input[sharedIdx].y = 0.0f;
          }

          sh_Input[sharedIdx + 256] = sh_Input[sharedIdx];
        }
      }
      else
      {
        if (threadIdx.y < 2)
        {
          if (threadIdx.y == 0)
            sh_Input[sharedIdx] = XInput[((Elementi + threadIdx.z)*Nspectra + k)*Nchannels + channelIdx];
          else
          {
            if (Elementj < Nelements)
              sh_Input[sharedIdx] = XInput[(Elementj*Nspectra + k)*Nchannels + channelIdx];
            else
            {
              sh_Input[sharedIdx].x = 0.0f;
              sh_Input[sharedIdx].y = 0.0f;
            }
          }
        }
      }
      /***  END Copy input data from global memory to shared memory  ***/


      __syncthreads();


      /***  BEGIN Multiply and accumulate  ***/

      tmp_input1 = sh_Input[threadIdx.y*64 + threadIdx.x];
      tmp_input2 = sh_Input[256 + threadIdx.z*64 + threadIdx.x];

      /*  Re  */
      tmp_output.x = fmaf(tmp_input1.x, tmp_input2.x, tmp_output.x);
      tmp_output.x = fmaf(tmp_input1.y, tmp_input2.y, tmp_output.x);

      /*  Im  */
      tmp_output.y = fmaf(tmp_input1.y,  tmp_input2.x, tmp_output.y);
      tmp_output.y = fmaf(tmp_input1.x, -tmp_input2.y, tmp_output.y);

      /***  END Multiply and accumulate  ***/
    }
    /***  END Loop through spectra, correlate and accumulate  ***/



    /***  BEGIN Write output to global memory  ***/
    if (Elementi < Nelements && Elementj < Nelements && Elementi <= Elementj)
    {
      tmp_output.x /= Nspectra;
      tmp_output.y /= Nspectra;

      k = ((long)(0.5f*Elementi*(2*Nelements - Elementi + 1)) + Elementj - Elementi) * Nchannels;

      FXOutput[k + channelIdx] = tmp_output;
    }
    /***  END Write output to global memory  ***/

  }
  /***  END Work for threads within Nchannels  ***/
}
