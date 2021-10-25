#ifndef HOMOGENEOUS_H_
#define HOMOGENEOUS_H_

#include <Eigen/Core>

Eigen::MatrixXd toHomogeneous(const Eigen::MatrixXd & mat)
{
  const auto rows = mat.rows();
  const auto cols = mat.cols();

  Eigen::MatrixXd out(rows + 1, cols);
  out.block(0, 0, rows, cols) = mat;
  for (int c = 0; c < cols; c++) {
    out(rows, c) = 1.0;
  }
  return out;
}

#endif
