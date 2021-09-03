#include <Eigen/Core>

Eigen::MatrixXf toHomogeneous(const Eigen::MatrixXf & mat)
{
  const auto rows = mat.rows();
  const auto cols = mat.cols();

  Eigen::MatrixXf out(rows + 1, cols);
  out.block(0, 0, rows, cols) = mat;
  for (int c = 0; c < cols; c++) {
    out(rows, c) = 1.0;
  }
  return out;
}
