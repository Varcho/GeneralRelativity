/*
  CUDA implementation of path tracing in curved spacetime.
  Copyright (C) Bill Varcho

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
*/

#include <iostream>
#include <ostream>
#include <string>
#include <vector>
#include <GLUT/glut.h>
#include <math.h>
#include <png.hpp>
#include <boost/program_options.hpp>
#include "/usr/local/cuda/include/cuda.h"
#include "/usr/local/cuda/include/math_functions.h"
#include "/usr/local/cuda/include/vector_types.h"
#include "/usr/local/cuda/include/vector_functions.h"
#include "/usr/local/cuda/include/device_launch_parameters.h"
#include "cutil_math.h"
#include "/usr/local/cuda/include/cuda_runtime.h"
#include "/usr/local/cuda/include/cuda_gl_interop.h"
#include "/usr/local/cuda/include/curand.h"
#include "/usr/local/cuda/include/curand_kernel.h"

const double PI = 3.1415926;
const double MAX_ITERS = 150; /* Number of iterations that rays are traced backwards
                              in time. This number was chosen after visual 
                              inspection of a couple of renders. */
const int WIDTH = 1*256;
const int HEIGHT = 1*128;

__device__ const double a = 50;
__device__ const double M = 150;
__device__ const double RHO = 200;
__device__ double L_INIT;     // Define the initial position of the observer
__device__ double THETA_INIT;
__device__ double PHI_INIT;

bool SAVE_IMG; // whether or not the final image should be saved in the out/ directory
bool RECORD_INTERIM; // whether or not the intermediate values of th raytraced paths should be recorded
int SAVE_ID; // id for saving the img (useful for image sequences)

GLuint vbo;
void *d_vbo_buffer = NULL;
// Declare the image pointer
float3 *imgptr;
// stores the intermediate values of the paths being raytraced.
// allows CUDA to take smaller stepsize and redraw, instead of just waiting for 
// everything to complete and then displaying.
double3* position_buffer;
double3* momentum_buffer;
double3* constants_buffer;
bool* intersection_buffer;

__device__ double r(const double l) {
  if (abs(l) > a) {
    double x = 2*(abs(l)-a)/(PI*M);
    return RHO+M*(x*atan(x)-.5*log(1+x*x));
  }
  return RHO;
}

__device__ double drdl(const double l) {
  // const double M = 150;
  return (abs(l)/l) * pow(abs(1-2*M/r(l)),0.5);
}

/*
 Derivatives used in the backward integration.
*/
__device__ double dldt(const double pl) {
  return pl;
}

__device__ double dthetadt(const double l, const double ptheta) {
  return ptheta / pow(r(l), 2.0);
}

__device__ double dphidt(const double l, const double theta, const double b) {
  return b / (pow(r(l),2.0)*pow(sin(theta),2.0));
}

__device__ double dpldt(const float l, const float B2) {
  return B2 * drdl(l) / pow(r(l), 3.0);
}

__device__ double dpthetadt(const double l, const double theta, const double b) {
  return pow(b/r(l),2.0)*cos(theta)/pow(sin(theta), 3.0);
}

// helper functions
__device__ double clamp(double x){ return x<0 ? 0 : x>1 ? 1 : x; }
__device__ double maxi(double a, double b){ return a<b ? b : a; }
__device__ double mini(double a, double b){ return a<b ? a : b; }
__device__ double absi(double x){ return x<0 ? -x : x; }


__device__ bool intersects_accretion(const double la, const double thetaa, const double phia,
                                    const double lb, const double thetab, const double phib) {
  // first check if one has theta above pi/2 and one below
  bool cross_horizon = (thetaa<=PI/2.0 && thetab >= PI/2.0) || (thetab<=PI/2.0 && thetaa >= PI/2.0);
  if (!cross_horizon) {
    return false;
  }

  // create quick and dirty estimation of radius of particle when it had the value of PI/2
  double drdtheta = (lb-la) / (thetab-thetaa);
  double r_approx = la + drdtheta * (PI/2.0 - thetaa);
  return r_approx >= 650 && r_approx <= 950;
}

/*---------------------------------------------------
  NUMERICAL INTEGRATORS
-----------------------------------------------------
  for this project I implemented a couple of various
numerical integration techniques to test on how well
they performed in the path tracing. My goal was to 
find the quickest integrator that didn't suffer from 
very big artifacts. */

/*
 Explicit implementation of Runge-Kutta 4th order
 integration. It is a generalized version of the 
 method that words for multiple dependent variables
 which are dependent and proper time.
*/
__device__ void ellis_rkf45(double &l, double &theta, double &phi, 
                            double &pl, double &ptheta, double &pphi, 
                            double b, double B2, bool &intersected) {

  double tol = .01;
  double h = -10.3; // h = 'dt' aka timestep
  double MINSTEP = -.1f;
  // Initialize tableau
  const double T[8][7] = {
    {0.f,     0.f,          0.f,          0.f,         0.f,         0.f,      0.f},
    {1/4.f,   1/4.f,        0.f,          0.f,         0.f,         0.f,      0.f},
    {3/8.f,   3/32.f,       9/32.f,       0.f,         0.f,         0.f,      0.f},
    {12/13.f, 1932/2197.f,  -7200/2197.f, 7296/2197.f, 0.f,         0.f,      0.f},
    {1.f,     439/219.f,    -8.f,         3680/513.f,  -845/4104.f, 0.f,      0.f},
    {1/2.f,   -8/27.f,      2.f,          -355/2565.f, 1859/4104.f, -11/40.f, 0.f},
    // and now the approximation coefficients
    {0.f, 25/216.f, 0.f, 1408/2565.f,  2197/4101.f,   -1/5.f,  0.f},
    {0.f, 16/135.f, 0.f, 6656/12825.f, 28561/56430.f, -9/50.f, 2/55.f},
  };

  for (int k = 0; k < 8; k++) {
    /*
      ai's correspond to l
      bi's correspond to theta
      ci's correspond to phi
      di's correspond to pl
      ei's correspond to ptheta
    */
    // compute the values for the 1st approximation
    // for all variables.. 
    double a1 = h*dldt(pl);
    double b1 = h*dthetadt(l,ptheta);
    double c1 = h*dphidt(l,theta,b);
    double d1 = h*dpldt(l,B2);
    double e1 = h*dpthetadt(l,theta,b);

    // .. and then the 2nd ... 
    double a2 = h*dldt(pl+T[1][1]*d1);
    double b2 = h*dthetadt(l+T[1][1]*a1,ptheta+T[1][1]*e1);
    double c2 = h*dphidt(l+T[1][1]*a1,theta+T[1][1]*b1,b);
    double d2 = h*dpldt(l+T[1][1]*a1,B2);
    double e2 = h*dpthetadt(l+T[1][1]*a1,theta+T[1][1]*b1,b);

    // .. and then the 3rd ... 
    double a3 = h*dldt(pl+T[2][1]*d1+T[2][2]*d2);
    double b3 = h*dthetadt(l+T[2][1]*a1+T[2][2]*a2,
                          ptheta+T[2][1]*e1+T[2][2]*e2);
    double c3 = h*dphidt(l+T[2][1]*a1+T[2][2]*a2,
                        theta+T[2][1]*b1+T[2][2]*b2,b);
    double d3 = h*dpldt(l+T[2][1]*a1+T[2][2]*a2,B2);
    double e3 = h*dpthetadt(l+T[2][1]*a1+T[2][2]*a2,
                           theta+T[2][1]*b1+T[2][2]*b2,b);

    // .. and then the 4th ... 
    double a4 = h*dldt(pl+T[3][1]*d1+T[3][2]*d2+T[3][3]*d3);
    double b4 = h*dthetadt(l+T[3][1]*a1+T[3][2]*a2+T[3][3]*a3,
                          ptheta+T[3][1]*e1+T[3][2]*e2+T[3][3]*e3);
    double c4 = h*dphidt(l+T[3][1]*a1+T[3][2]*a2+T[3][3]*a3,
                        theta+T[3][1]*b1+T[3][2]*b2+T[3][3]*b3,
                        b);
    double d4 = h*dpldt(l+T[3][1]*a1+T[3][2]*a2+T[3][3]*a3,B2);
    double e4 = h*dpthetadt(l+T[3][1]*a1+T[3][2]*a2+T[3][3]*a3,
                           theta+T[3][1]*b1+T[3][2]*b2+T[3][3]*b3,
                           b);

    // .. and then the 5th ... 
    double a5 = h*dldt(pl+T[4][1]*d1+T[4][2]*d2+T[4][3]*d3+T[4][4]*d4);
    double b5 = h*dthetadt(l+T[4][1]*a1+T[4][2]*a2+T[4][3]*a3+T[4][4]*a4,
                          ptheta+T[4][1]*e1+T[4][2]*e2+T[4][3]*e3+T[4][4]*e4);
    double c5 = h*dphidt(l+T[4][1]*a1+T[4][2]*a2+T[4][3]*a3+T[4][4]*a4,
                        theta+T[4][1]*b1+T[4][2]*b2+T[4][3]*b3+T[4][4]*b4,
                        b);
    double d5 = h*dpldt(l+T[4][1]*a1+T[4][2]*a2+T[4][3]*a3+T[4][4]*a4,B2);
    double e5 = h*dpthetadt(l+T[4][1]*a1+T[4][2]*a2+T[4][3]*a3+T[4][4]*a4,
                           theta+T[4][1]*b1+T[4][2]*b2+T[4][3]*b3+T[4][4]*b4,
                           b);

    // .. and then the 6th ... 
    double a6 = h*dldt(pl+T[5][1]*d1+T[5][2]*d2+T[5][3]*d3+T[5][4]*d4+T[5][5]*d5);
    double b6 = h*dthetadt(l+T[5][1]*a1+T[5][2]*a2+T[5][3]*a3+T[5][4]*a4+T[5][5]*a5,
                          ptheta+T[5][1]*e1+T[5][2]*e2+T[5][3]*e3+T[5][4]*e4+T[5][5]*e5);
    double c6 = h*dphidt(l+T[5][1]*a1+T[5][2]*a2+T[5][3]*a3+T[5][4]*a4+T[5][5]*a5,
                        theta+T[5][1]*b1+T[5][2]*b2+T[5][3]*b3+T[5][4]*b4+T[5][5]*b5,
                        b);
    double d6 = h*dpldt(l+T[5][1]*a1+T[5][2]*a2+T[5][3]*a3+T[5][4]*a4+T[5][5]*a5,B2);
    double e6 = h*dpthetadt(l+T[5][1]*a1+T[5][2]*a2+T[5][3]*a3+T[5][4]*a4+T[5][5]*a5,
                           theta+T[5][1]*b1+T[5][2]*b2+T[5][3]*b3+T[5][4]*b4+T[5][5]*b5,
                           b);

    // create two approximations of the value at the next timestep
    // approx. with lower order
    double ly      = l+T[6][1]*a1+T[6][3]*a3+T[6][4]*a4+T[6][5]*a5;
    double thetay  = theta+T[6][1]*b1+T[6][3]*b3+T[6][4]*b4+T[6][5]*b5;
    double phiy    = phi+T[6][1]*c1+T[6][3]*c3+T[6][4]*c4+T[6][5]*c5;
    double ply     = pl+T[6][1]*d1+T[6][3]*d3+T[6][4]*d4+T[6][5]*d5;
    double pthetay = ptheta+T[6][1]*e1+T[6][3]*e3+T[6][4]*e4+T[6][5]*e5;
    // higher order approx
    double lz      = l+T[7][1]*a1+T[7][3]*a3+T[7][4]*a4+T[7][5]*a5+T[7][6]*a6;
    double thetaz  = theta+T[7][1]*b1+T[7][3]*b3+T[7][4]*b4+T[7][5]*b5+T[7][6]*b6;
    double phiz    = phi+T[7][1]*c1+T[7][3]*c3+T[7][4]*c4+T[7][5]*c5+T[7][6]*c6;
    double plz     = pl+T[7][1]*d1+T[7][3]*d3+T[7][4]*d4+T[7][5]*d5+T[7][6]*d6;
    double pthetaz = ptheta+T[7][1]*e1+T[7][3]*e3+T[7][4]*e4+T[7][5]*e5+T[7][6]*e6;

    // use the two approximations (and their differences) to compute the optimal timestep
    // if (within_tol())
    double mdiff = absi(ly-lz);
    mdiff = maxi(absi(thetay-thetaz),mdiff);
    mdiff = maxi(absi(phiy-phiz),mdiff);
    mdiff = maxi(absi(ply-plz),mdiff);
    mdiff = maxi(absi(pthetay-pthetaz),mdiff);
    if (mdiff < 0.000001) {
      mdiff = 0.000001;
    }

    double hold = h;
    // actually set the step size
    h *= .84 * pow(absi(tol*h/mdiff), .25f);

    if (std::abs(l) > 10000000) {
      k = 100000000; // break loop
    } else if (intersects_accretion(l, theta, phi, lz, thetaz, phiz)) {
      intersected = true;
      k = 100000000;
    } else {
      if (mdiff < tol) { 
        l = lz;
        theta = thetaz;
        phi = phiz;
        pl = plz;
        ptheta = pthetaz;
      } else {
        h = hold/2.0;
      }
    }
    h = mini(h, MINSTEP);
  }
}

// /*
//  Explicit implementation of Runge-Kutta 4th order
//  integration. It is a generalized version of the 
//  method that words for multiple dependent variables
//  which are dependent and proper time.
// */
// __device__ void ellis_rk4(float &l, float &theta, float &phi, 
//                           float &pl, float &ptheta, float &pphi, 
//                           float b, float B2) {
//   /* This particular method uses the following integration tableau
//      .0|
//      .5| .5
//      .5| .0  .5
//     1.0| .0  .0  1
//     ---------------
//          .16 .33 .33 .16 */  

//   float h = -10.3; // h = 'dt' aka timestep
//   for (int k = 0; k < 5; k++) {
//     /*
//       ai's correspond to l
//       bi's correspond to theta
//       ci's correspond to phi
//       di's correspond to pl
//       ei's correspond to ptheta
//     */
//     // compute the values for the 0th approximation
//     // for all variables.. 
//     float a0 = h*dldt(pl);
//     float b0 = h*dthetadt(l,ptheta);
//     float c0 = h*dphidt(l,theta,b);
//     float d0 = h*dpldt(l,B2);
//     float e0 = h*dpthetadt(l,theta,b);

//     // .. and then the 1st ... 
//     float a1 = h*dldt(pl+.5*d0);
//     float b1 = h*dthetadt(l+.5*a0,ptheta+.5*e0);
//     float c1 = h*dphidt(l+.5*a0,theta+.5*b0,b);
//     float d1 = h*dpldt(l+.5*a0,B2);
//     float e1 = h*dpthetadt(l+.5*a0,theta+.5*b0,b);
    
//     // ... and 2nd ...
//     float a2 = h*dldt(pl+.5*d1);
//     float b2 = h*dthetadt(l+.5*a1,ptheta+.5*e1);
//     float c2 = h*dphidt(l+.5*a1,theta+.5*b1,b);
//     float d2 = h*dpldt(l+.5*a1,B2);
//     float e2 = h*dpthetadt(l+.5*a1,theta+.5*b1,b);
    
//     // ... and finally the 3rd
//     float a3 = h*dldt(pl+d2);
//     float b3 = h*dthetadt(l+a2,ptheta+e2);
//     float c3 = h*dphidt(l+a2,theta+b2,b);
//     float d3 = h*dpldt(l+a2,B2);
//     float e3 = h*dpthetadt(l+a2,theta+b2,b);

//     // Then use these values to compute the derivative approximation
//     // used in the backwards time step
//     l      = l+(a0+2*a1+2*a2+a3)/6.0;
//     theta  = theta+(b0+2*b1+2*b2+b3)/6.0;
//     phi    = phi+(c0+2*c1+2*c2+c3)/6.0;
//     pl     = pl+(d0+2*d1+2*d2+d3)/6.0;
//     ptheta = ptheta+(e0+2*e1+2*e2+e3)/6.0;
//   } 
// }

// __device__ void ellis_euler(float &l, float &theta, float &phi, 
//                             float &pl, float &ptheta, float &pphi, 
//                             float b, float B2) {
//   float h = -5.3; // h = 'dt' aka timestep
//   for (int k = 0; k < 10; k++) {
//     /*
//       ai's correspond to l
//       bi's correspond to theta
//       ci's correspond to phi
//       di's correspond to pl
//       ei's correspond to ptheta
//     */
//     // compute the values for the 0th approximation
//     // for all variables.. 
//     float a0 = h*dldt(pl);
//     float b0 = h*dthetadt(l,ptheta);
//     float c0 = h*dphidt(l,theta,b);
//     float d0 = h*dpldt(l,B2);
//     float e0 = h*dpthetadt(l,theta,b);

//     // Then use these values to compute the derivative approximation
//     // used in the backwards time step
//     l      = l+a0;
//     theta  = theta+b0;
//     phi    = phi+c0;
//     pl     = pl+d0;
//     ptheta = ptheta+e0;
//   } 
// }


__host__ __device__ double floatmod(double x, double n) {
  int mult = 0.0;
  int opt = x/(-n)+1.0f;
  if (opt > mult) {
    mult = opt;
  }
  x += mult * n;
  int dv = (int) (x / n);
  x -= dv*n;
  return x;
}

// Helper data structure for converting colors into openGL friendly form
union Colour  // 4 bytes = 4 chars = 1 float
{
  float c;
  uchar4 components;
};

/*
  float3 *output: imagebuffer that will display the rendered image
*/
__global__ void render_kernel(float3 *output, double3 *pbuff, double3 *mbuff, double3 *cbuff, bool *ibuff, 
                              int width, int height) {
  // create a CUDA thread for every pixel in the image buffer
  // and assign it a unique id
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  // init params for integration
  double lc,thetac,phic,pli,pthetai,pphii,b,B2,timer;
  // get the position of the pixel in memory
  int i = (height - y - 1)*width + x;
  // get the current position, momentum, and constants of motion for the
  // particle path
  double3 position = pbuff[i];
  double3 momentum = mbuff[i];
  double3 constants = cbuff[i];

  Colour fcolour;
  float3 colour;
  if (constants.z < 1.0) {
    /*-------------------------------------------------------------
     Do the necessary setup for the light path on initialization.
    ---------------------------------------------------------------*/
    // at first hardcode the initial position of the viewer
        lc = L_INIT;
    thetac = THETA_INIT;
      phic = PHI_INIT;

    // initialize ray parameters based on pixel
    // float thetacs = (PI*(y)) / ((float) height);
    // float phics =  (2*PI*x) / ((float) width);
    double thetacs = (PI*(y)) / ((double) height);
    double phics = (2*PI*x) / ((double) width);

    // compute the components of the ray vector
    double Nx = sin(thetacs) * cos(phics),
           Ny = sin(thetacs) * sin(phics),
           Nz = cos(thetacs);
    
    // now get the canonical momentum components
        pli = -Nx,
    pthetai = r(lc) * Nz,
      pphii = -r(lc) * sin(thetac) * Ny;

    // ... and finally the constants of motion
     b = pphii,
    B2 = pow(r(lc), 2.0) * (pow(Nz,2.0)+pow(Ny,2.0));
    timer = 2.0;
  } else {
    //  extract the data from the accumulation
     lc =  position.x,  thetac =  position.y,  phic = position.z;
    pli =  momentum.x, pthetai =  momentum.y, pphii = momentum.z;
      b = constants.x,      B2 = constants.y, timer = constants.z;
  }

  bool intersected = ibuff[i];
  ellis_rkf45(lc,thetac,phic,pli,pthetai,pphii,b,B2,intersected);
  pbuff[i] = make_double3(lc,thetac,phic);
  mbuff[i] = make_double3(pli,pthetai,pphii);
  cbuff[i] = make_double3(b,B2,timer+.1);
  ibuff[i] = intersected;

  // modify input so that it can be displayed nicely
  thetac = floatmod(thetac, (double)PI) / ((double)PI);
  phic = floatmod(phic, (double)2.0f*PI)/ ((double)2*PI);

  colour = make_float3((float)phic, (float)thetac, 0.0);
  if (lc < 0.0) {
    colour = make_float3((float)phic, (float)thetac, 1.0);
  }
  if (intersected) {
    colour = make_float3((float)phic, (float)thetac, 0.5);
  }

  fcolour.components = make_uchar4(
    (unsigned char)(powf(colour.x, 1.0f) * 255), 
    (unsigned char)(powf(colour.y, 1.0f) * 255), 
    (unsigned char)(powf(colour.z, 1.0f) * 255), 
    1
  );
  output[i] = make_float3(x, y, fcolour.c);
}


// class for writing the output image to a file
class pixel_generator : public png::generator<png::gray_pixel_1, pixel_generator> {
public:
  pixel_generator(size_t width, size_t height)
      : png::generator< png::gray_pixel_1, pixel_generator >(width, height), m_row(width) {
    for (size_t i = 0; i < m_row.size(); ++i) {
      m_row[i] = i > m_row.size() / 2 ? 1 : 0;
    }
  }

  png::byte* get_next_row(size_t /*pos*/) {
    size_t i = std::rand() % m_row.size();
    size_t j = std::rand() % m_row.size();
    png::gray_pixel_1 t = m_row[i];
    m_row[i] = m_row[j];
    m_row[j] = t;
    return reinterpret_cast< png::byte* >(row_traits::get_data(m_row));
  }

private:
  typedef png::packed_pixel_row< png::gray_pixel_1 > row;
  typedef png::row_traits< row > row_traits;
  row m_row;
};

// write an image based on the current contents of 
// the position and intersection buffers.
void writeImage(std::string filepath) {
  // first copy the position device array that lives in the CUDA world
  // over to an array that we can access on the host device...
  int NUM_BYTES = WIDTH * HEIGHT * sizeof(double3);
  double3 *posHostArray= (double3*)malloc(NUM_BYTES);
  double3 *posDeviceArray = position_buffer;
  memset(posHostArray,0,NUM_BYTES);
  cudaMemcpy(posHostArray,posDeviceArray,NUM_BYTES,cudaMemcpyDeviceToHost);

  // ... and then do the same for the intersection array ...
  NUM_BYTES = WIDTH * HEIGHT * sizeof(bool);
  bool *intHostArray= (bool*)malloc(NUM_BYTES);
  bool *intDeviceArray = intersection_buffer;
  memset(intHostArray,0,NUM_BYTES);
  cudaMemcpy(intHostArray,intDeviceArray,NUM_BYTES,cudaMemcpyDeviceToHost);

  // write the array as a png image
  png::image< png::rgb_pixel > image(WIDTH, HEIGHT);
  for (png::uint_32 y = 0; y < image.get_height(); y++) {
      for (png::uint_32 x = 0; x < image.get_width(); x++) {
          int i = (HEIGHT - y - 1)*WIDTH + x; // TODO(bill) move to function
          double3 position = posHostArray[i];
          bool intersected = intHostArray[i];
          double lc = position.x, thetac =  position.y,  phic = position.z;
          thetac = floatmod(thetac, (double)PI) / ((double)PI);
          phic = floatmod(phic, (double)2.0f*PI)/ ((double)2*PI);

          float3 colour = make_float3((float)phic, (float)thetac, 0.0);
          if (intersected) {
            colour.z = 0.5;
          } else if (lc < 0.0) {
            colour.z = 1.0;
          }
          image[y][x] = png::rgb_pixel((powf(colour.x, 1.0f) * 255),
                                       (powf(colour.y, 1.0f) * 255), 
                                       (powf(colour.z, 1.0f) * 255));
      }
  }
  image.write(filepath);
}

void writeAndReturn(void) {
  if (SAVE_IMG) {
    char buff[10];
    snprintf(buff, sizeof(buff), "%05d", SAVE_ID);
    std::string num = buff;
    std::string filename = "out/outimg_" + num + ".png";
    writeImage(filename);
  }
  exit(EXIT_SUCCESS);
}

void display(void) {
  // TODO: update the height and buffer depending on the size
  cudaThreadSynchronize();
  // allow CUDA to acces the vbo
  cudaGLMapBufferObject((void**)&imgptr, vbo);
  // clear the buffer
  glClear(GL_COLOR_BUFFER_BIT);

  // specify the cuda specific block and grid size
  dim3 block(32,32,1);
  dim3 grid(WIDTH / block.x, HEIGHT / block.y, 1);

  // now launch the kernel
  render_kernel <<< grid, block >>> (imgptr, position_buffer, 
                                             momentum_buffer, 
                                             constants_buffer, 
                                             intersection_buffer, 
                                             WIDTH, HEIGHT);
  cudaThreadSynchronize();
  // now unmap the buffer
  cudaGLUnmapBufferObject(vbo);
  // and display
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glVertexPointer(2, GL_FLOAT, 12, 0);
  glColorPointer(4, GL_UNSIGNED_BYTE, 12, (GLvoid*)8);
  glEnableClientState(GL_VERTEX_ARRAY);
  glEnableClientState(GL_COLOR_ARRAY);
  glDrawArrays(GL_POINTS, 0, WIDTH*HEIGHT);
  glDisableClientState(GL_VERTEX_ARRAY);
  glutSwapBuffers();
}

void createVBO(GLuint* vbo) {
  glGenBuffers(1, vbo);
  glBindBuffer(GL_ARRAY_BUFFER, *vbo);
  // initialize the vbo
  unsigned int size = WIDTH * HEIGHT * sizeof(float3);
  glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  // and finally register the VBO with CUDA
  cudaGLRegisterBufferObject(*vbo);
}

void drawLoop(int iter) {
  glutPostRedisplay();

  // draw the progress bar based on the current iteration
  float progress = iter / (float) MAX_ITERS;
  int barWidth = 70;
  std::cout << "[";
  int pos = barWidth * progress;
  for (int i = 0; i < barWidth; ++i) {
      if (i < pos) std::cout << "=";
      else if (i == pos) std::cout << ">";
      else std::cout << " ";
  }
  std::cout << "] " << int(progress * 100.0) << " %\r";
  std::cout.flush();

  // Continue drawing if less than the max iterations
  if (iter < MAX_ITERS) {
    if (RECORD_INTERIM) {
      char buff[10];
      snprintf(buff, sizeof(buff), "%05d", iter);
      std::string num = buff;
      std::string filename = "record/img_" + num + ".png";
      writeImage(filename);
    }
    glutTimerFunc(200, drawLoop, iter+1);
  } else {
    writeAndReturn();
  }
}

int main(int argc, char** argv) {

  // parse out the program options specified by the user 
  namespace po = boost::program_options;
  po::options_description desc("Allowed Options");
  desc.add_options()
    ("save,s", po::bool_switch()->default_value(false), "save the final image")
    ("record,r", po::bool_switch()->default_value(false), "record intermediate images")
    ("num,n", po::value<int>()->default_value(0), "temporal id of the image to be saved")
    ("li,l", po::value<double>()->default_value(1500.0), "radius of FIDO")
    ("thetai,t", po::value<double>()->default_value(PI/2.0f), "angle from the pole of FIDO")
    ("phii,p", po::value<double>()->default_value(0.6f), "angle around pole of FIDO");
  // TODO: options for height and width of the final image
  //       options for raytrace time

  po::variables_map vm;
  po::store(po::parse_command_line(argc,argv,desc),vm);
  po::notify(vm);
  SAVE_IMG = vm.count("save") ? vm["save"].as<bool>() : false;
  RECORD_INTERIM = vm.count("record") ? vm["record"].as<bool>() : false;
  SAVE_ID =  vm["num"].as<int>();

  // set the initial position of the observer based on the 
  // command line arguments
  double L_INIT_HOST     = vm["li"].as<double>();
  double THETA_INIT_HOST = vm["thetai"].as<double>() +.4;
  double PHI_INIT_HOST   = vm["phii"].as<double>();
  cudaMemcpyToSymbol(L_INIT, &L_INIT_HOST, sizeof(double));
  cudaMemcpyToSymbol(THETA_INIT, &THETA_INIT_HOST, sizeof(double));
  cudaMemcpyToSymbol(PHI_INIT, &PHI_INIT_HOST, sizeof(double));

  // initialize and allocate the buffers....
  int NUM_BYTES = WIDTH * HEIGHT  * sizeof(double3);
  std::cout << "Allocating buffers..." << std::endl;
  cudaMalloc(&position_buffer, NUM_BYTES);
  cudaMalloc(&momentum_buffer, NUM_BYTES);
  cudaMalloc(&constants_buffer, NUM_BYTES);
  cudaMalloc(&intersection_buffer, WIDTH * HEIGHT * sizeof(bool));
  std::cout << "Filling buffers..." << std::endl;
  cudaMemset(position_buffer, 0, NUM_BYTES);
  cudaMemset(momentum_buffer, 0, NUM_BYTES);
  cudaMemset(constants_buffer, 0, NUM_BYTES);
  cudaMemset(intersection_buffer, false, WIDTH * HEIGHT  * sizeof(bool));

  // init glut for OpenGL viewport
  glutInit(&argc, argv);
  // specify the display mode to be RGB and single buffering
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
  // specify the initial window position
  glutInitWindowPosition(100, 100);
  // specify the initial window size
  glutInitWindowSize(WIDTH, HEIGHT);
  // create the window and set title
  glutCreateWindow("Ellis GPU");
  // init OpenGL
  glClearColor(0.0, 0.0, 0.0, 0.0);
  glMatrixMode(GL_PROJECTION);
  gluOrtho2D(0.0, WIDTH, 0.0, HEIGHT);
  // register callback function to display graphics:
  glutDisplayFunc(display);
  // init redrawing
  drawLoop(0);
  createVBO(&vbo);
  glutMainLoop();
  // free the CUDA memory
  cudaFree(position_buffer);
  cudaFree(momentum_buffer);
  cudaFree(constants_buffer);
  cudaFree(imgptr);
}