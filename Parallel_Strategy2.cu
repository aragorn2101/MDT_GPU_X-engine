/*
 *  CUDA kernel for X-engine implementing parallel strategy 2.
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
 *  X-engine kernel -- Parallel Strategy 2
 *
 *  Nelements: number of elements in interferometer,
 *  Nchannels: number of channels in 1 spectrum,
 *  Nspectra: total number of spectra within 1 integration time,
 *  XInput: input array for X-tengine coming directly from F-engine,
 *  FXOutput: output array for the whole FX operation.
 *
 *  The thread grid is setup using the following:
 *
 *  NumThreadx = (int)( MaxShMemPerBlock / ((sizeof(cufftComplex) + sizeof(float2))*Nelements) );
 *  if (NumThreadx > MaxThreadsPerBlock)
 *    NumThreadx = MaxThreadsPerBlock;
 *
 *  NumThready = 1;
 *  NumThreadz = 1;
 *
 *  NumBlockx = (Nchannels/NumThreadx) + ((Nchannels%NumThreadx != 0) ? 1 : 0);
 *  NumBlocky = Nelements/2 + 1;
 *  NumBlockz = 1;
 *
 *  ShMemPerBlock = (sizeof(cufftComplex) + sizeof(float2)) * Nelements * NumThreadx;
 *
 *  And it is lauched with the following call:
 *  XEngine<<< dim3(NumBlockx,NumBlocky,NumBlockz), dim3(NumThreadx,NumThready,NumThreadz), ShMemPerBlock >>>(Nelements, Nchannels, Nspectra, d_XInput, d_FXOutput);
 *
 */
__global__ void XEngine(int Nelements, int Nchannels, int Nspectra, cufftComplex *XInput, float2 *FXOutput)
{
  extern __shared__ cufftComplex sh_Total[];

  cufftComplex *sh_Input = sh_Total;
  float2 *sh_Output = (float2 *)&sh_Input[Nelements * blockDim.x];

  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int k;
  long stride;

  cufftComplex tmp_input1, tmp_input2;
  float2 tmp_output;

  if (i < Nchannels)
  {
    /*  Initialize shared memory output array  */
    for ( i=0 ; i<Nelements ; i++ )
    {
      sh_Output[i*blockDim.x + threadIdx.x].x = 0.0f;
      sh_Output[i*blockDim.x + threadIdx.x].y = 0.0f;
    }


    /***  BEGIN LOOP over spectra within integration time  ***/
    for ( k=0 ; k<Nspectra ; k++ )
    {
      /*  Read input data into shared memory  */
      stride = k*Nchannels + blockIdx.x*blockDim.x + threadIdx.x;
      for ( i=blockIdx.y ; i<Nelements ; i++ )
        sh_Input[i*blockDim.x + threadIdx.x] = XInput[i*Nspectra*Nchannels + stride];


      /***  BEGIN Correlation of row corresponding to blockIdx.y  ***/
      stride = blockIdx.y*blockDim.x + threadIdx.x;


      /*  Auto-correlation  */
      tmp_input1 = sh_Input[stride];
      tmp_output = sh_Output[threadIdx.x];

      tmp_output.x = fmaf(tmp_input1.x , tmp_input1.x, tmp_output.x);
      tmp_output.x = fmaf(tmp_input1.y , tmp_input1.y, tmp_output.x);

      sh_Output[threadIdx.x] = tmp_output;


      /*  Cross-correlation  */
      for ( i=1 ; i<Nelements-blockIdx.y ; i++ )
      {
        tmp_input2 = sh_Input[i*blockDim.x + stride];
        tmp_output = sh_Output[i*blockDim.x + threadIdx.x];

        // Re
        tmp_output.x = fmaf(tmp_input1.x, tmp_input2.x, tmp_output.x);
        tmp_output.x = fmaf(tmp_input1.y, tmp_input2.y, tmp_output.x);

        // Im
        tmp_output.y = fmaf(tmp_input1.y,  tmp_input2.x, tmp_output.y);
        tmp_output.y = fmaf(tmp_input1.x, -tmp_input2.y, tmp_output.y);

        sh_Output[i*blockDim.x + threadIdx.x] = tmp_output;
      }
      /***  END Correlation of row corresponding to blockIdx.y  ***/



      /***  BEGIN Cross-multiplication of folded row (Nelements - blockIdx.y)  ***/
      if (blockIdx.y != 0 && blockIdx.y != Nelements - blockIdx.y)
      {
        stride = (Nelements - blockIdx.y)*blockDim.x + threadIdx.x;


        /*  Auto-correlation  */
        tmp_input1 = sh_Input[stride];
        tmp_output = sh_Output[stride];

        tmp_output.x = fmaf(tmp_input1.x, tmp_input1.x, tmp_output.x);
        tmp_output.x = fmaf(tmp_input1.y, tmp_input1.y, tmp_output.x);

        sh_Output[stride] = tmp_output;


        /*  Cross-correlation  */
        for ( i=1 ; i<blockIdx.y ; i++ )
        {
          tmp_input2 = sh_Input[i*blockDim.x + stride];
          tmp_output = sh_Output[i*blockDim.x + stride];

          // Re
          tmp_output.x = fmaf(tmp_input1.x, tmp_input2.x, tmp_output.x);
          tmp_output.x = fmaf(tmp_input1.y, tmp_input2.y, tmp_output.x);

          // Im
          tmp_output.y = fmaf(tmp_input1.y,  tmp_input2.x, tmp_output.y);
          tmp_output.y = fmaf(tmp_input1.x, -tmp_input2.y, tmp_output.y);

          sh_Output[i*blockDim.x + stride] = tmp_output;
        }
      }
      /***  END Cross-multiplication of folded row (Nelements - blockIdx.y)  ***/

    }
    /***  END LOOP over spectra within integration time  ***/



    /***  BEGIN Write output to global memory array  */

    /*  Upper part of the matrix  */
    if (blockIdx.y == 0)
      stride = blockIdx.x*blockDim.x + threadIdx.x;
    else
      stride = (long)( 0.5f*blockIdx.y * (2.0f*Nelements - blockIdx.y + 1.0f) * Nchannels ) + blockIdx.x*blockDim.x + threadIdx.x;

    for ( i=0 ; i<Nelements-blockIdx.y ; i++ )
    {
      tmp_output = sh_Output[i*blockDim.x + threadIdx.x];

      tmp_output.x /= Nspectra;
      tmp_output.y /= Nspectra;

      FXOutput[i*Nchannels + stride] = tmp_output;
    }

    /*  Folded part of the matrix  */
    if (blockIdx.y != 0 && blockIdx.y != Nelements - blockIdx.y)
    {
      stride = (long)( 0.5f*(Nelements - blockIdx.y) * (Nelements + blockIdx.y + 1.0f) * Nchannels ) + blockIdx.x*blockDim.x + threadIdx.x;

      for ( i=0 ; i<blockIdx.y ; i++ )
      {
        tmp_output = sh_Output[(Nelements - blockIdx.y + i)*blockDim.x + threadIdx.x];

        tmp_output.x /= Nspectra;
        tmp_output.y /= Nspectra;

        FXOutput[i*Nchannels + stride] = tmp_output;
      }
    }

    /***  END Write output to global memory array  */

  }
}
