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



/*
 *  X-engine kernel -- Parallel Strategy 2
 *
 *  Nelements: number of elements in interferometer,
 *  Nchannels: number of channels in 1 spectrum,
 *  Nspectra: total number of spectra within 1 integration time,
 *  XInput: input array for X-tengine coming directly from F-engine,
 *  FXOutput: output array for the whole FX operation.
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
