#include <fftw3.h>

/*
 *  X-engine function -- Serial Implementation
 *
 *  Nelements: number of elements,
 *  Nchannels: number of channels in 1 spectrum,
 *  Nspectra: number of spectra to be accumulated over
 *            integration time,
 *  XInput: input signal array in quadrature sampling format,
 *  FXOutput: output spectra for each baseline.
 *
 */
void XEngine(int Nelements, int Nchannels, int Nspectra,
                   fftwf_complex *XInput, fftwf_complex *FXOutput)
{
  int i, j, k, l;
  long stride1, stride2;


  for ( k=0 ; k<Nelements ; k++ )
    for ( j=k ; j<Nelements ; j++ )
    {
      /*  Stride for output array  */
      stride1 = ((int)( 0.5f*k*(2.0f*Nelements - k + 1.0f) + j - k)) * Nchannels;

      for ( i=0 ; i<Nchannels ; i++ )  // loop on frequency channel
      {
        FXOutput[stride1 + i][0] = 0.0f;
        FXOutput[stride1 + i][1] = 0.0f;

        for ( l=0 ; l<Nspectra ; l++ )  // loop on spectra (within integration time)
        {
          stride2 = (k*Nspectra + l ) * Nchannels;
          stride3 = (j*Nspectra + l ) * Nchannels;

          FXOutput[stride1 + i][0] += XInput[stride2 + i][0]*XInput[stride3 + i][0] + XInput[stride2 + i][1]*XInput[stride3 + i][1];
          FXOutput[stride1 + i][1] += XInput[stride2 + i][1]*XInput[stride3 + i][0] - XInput[stride2 + i][0]*XInput[stride3 + i][1];
        }

        FXOutput[stride1 + i][0] /= Nspectra;
        FXOutput[stride1 + i][1] /= Nspectra;
      }
    }
}
