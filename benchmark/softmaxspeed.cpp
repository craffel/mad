// Compile:
// g++ -I /usr/include/eigen3/ softmaxspeed.cpp -o softmaxspeed
// Run:
// ./softmaxspeed 10 20 30 40

#include <iostream>
#include <Eigen/Dense>
#include <ctime>

using namespace Eigen;
using namespace std;

int main(int argc, char *argv[])
{
  if ( argc != 5 ) {
    cout << "Usage: softmaxspeed T U N n_trials" << endl;
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
    for (int i = 0; i < U; i++) {
      // Compute dot product of the i'th row of s against all rows of x
      energy = x * s.row(i).transpose();
      // Compute softmax(energy)_n = exp(energy[n] - max(energy))/sum(exp(energy))
      energy_max = energy.maxCoeff();
      energy -= energy_max*VectorXd::Ones(energy.size());
      attention = energy.array().exp();
      attention /= attention.array().sum();
      // Compute weighted sum of each entry in x
      context = (x.array().colwise() * attention.array()).colwise().sum();
    }
  }
  clock_t end = clock();
  cout << double(end - begin) / CLOCKS_PER_SEC << endl;
}
