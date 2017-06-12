// Compile:
// g++ -I /usr/include/eigen3/ monotonicspeed.cpp -o monotonicspeed
// Run:
// ./monotonicspeed 10 20 30 40

#include <iostream>
#include <Eigen/Dense>
#include <ctime>

using namespace Eigen;
using namespace std;

int main(int argc, char *argv[])
{
  if ( argc != 5 ) {
    cout << "Usage: monotonicspeed T U N n_trials" << endl;
    return 0;
  }

  // Number of timesteps in the input
  const int T = atoi(argv[1]);
  // Number of output timesteps
  const int U = atoi(argv[2]);
  // Dimensionality of the input/output
  const int N = atoi(argv[3]);
  // Number of trials to run for timing
  const int n_trials = atoi(argv[4]);

  // Initializers
  double energy_max;
  VectorXd energy;
  VectorXd attention;
  VectorXd context;
  MatrixXd x = MatrixXd::Random(T, N);
  MatrixXd s = MatrixXd::Random(U, N);

  // Time it
  clock_t begin = clock();
  for (int trial = 0; trial < n_trials; trial++) {
    int j = 0;
    for (int i = 0; i < U; i++) {
      // Compute dot product of the i'th row of s against j'th row of x
      // Keep incrementing j until the result is > 0
      while (j < T - 1 && x.row(j) * s.row(i).transpose() < 0.) {
        j++;
      }
      context = x.row(j);
    }
  }
  clock_t end = clock();
  cout << double(end - begin) / CLOCKS_PER_SEC << endl;
}
