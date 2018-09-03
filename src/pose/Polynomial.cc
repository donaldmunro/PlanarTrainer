// Copyright (C) 2013 The Regents of the University of California (Regents).
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//
//     * Neither the name of The Regents or University of California nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Please contact the author of this library if you have any questions.
// Author: Chris Sweeney (cmsweeney@cs.ucsb.edu)

#include "Polynomial.h"

#include <complex>
#include <cmath>

int SolveQuadratic(const double a, const double b, const double c,
                   std::complex<double> *roots)
{
   // If the equation is actually linear.
   if (a == 0.0)
   {
      roots[0] = -1.0 * c / b;
      return 1;
   }

   const double D = b * b - 4 * a * c;
   const double sqrt_D = std::sqrt(std::abs(D));

   // Real roots.
   if (D >= 0)
   {
      // Stable quadratic roots according to BKP Horn.
      // http://people.csail.mit.edu/bkph/articles/Quadratics.pdf
      if (b >= 0)
      {
         roots[0] = (-b - sqrt_D) / (2.0 * a);
         roots[1] = (2.0 * c) / (-b - sqrt_D);
      }
      else
      {
         roots[0] = (2.0 * c) / (-b + sqrt_D);
         roots[1] = (-b + sqrt_D) / (2.0 * a);
      }
      return 2;
   }

   // Use the normal quadratic formula for the complex case.
   roots[0].real(-b / (2.0 * a));
   roots[1].real(-b / (2.0 * a));
   roots[0].imag(sqrt_D / (2.0 * a));
   roots[1].imag(-sqrt_D / (2.0 * a));
   return 2;
}

// Provides solutions to the equation a*x^2 + b*x + c = 0.
int SolveQuadraticReals(const double a, const double b, const double c,
                        double *roots)
{
   std::complex<double> complex_roots[2];
   int num_complex_solutions = SolveQuadratic(a, b, c, complex_roots);
   int num_real_solutions = 0;
   for (int i = 0; i < num_complex_solutions; i++)
   {
      roots[num_real_solutions++] = complex_roots[i].real();
   }
   return num_real_solutions;
}

int SolveQuadraticReals(const double a, const double b, const double c,
                        const double tolerance, double *roots)
{
   std::complex<double> complex_roots[2];
   int num_complex_solutions = SolveQuadratic(a, b, c, complex_roots);
   int num_real_solutions = 0;
   for (int i = 0; i < num_complex_solutions; i++)
   {
      if (std::abs(complex_roots[i].imag()) < tolerance)
      {
         roots[num_real_solutions++] = complex_roots[i].real();
      }
   }
   return num_real_solutions;
}