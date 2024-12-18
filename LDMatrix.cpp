#include <iostream>
#include <fstream>
#include <string>
#include <map>

#include <armadillo>
#include <sstream>

#include "LDMatrix.h"
using namespace std;

LDMatrix::LDMatrix(const string& filename) { // instance만들면 바로 load.

	// [0] load하기.

	// [0-1] fstream 열기
	std::ifstream file(filename);

	if (!file.is_open()) {
		std::cerr << "Error: Unable to open file " << filename << std::endl;
		exit(1);
	}

	// [0-2] 첫 번째 줄: 헤더 읽기
	std::string line;

	if (std::getline(file, line)) {

		std::istringstream iss(line);
		std::string header;

		for (M = 0; iss >> header; ++M) { // `M` is incremented here.

			col_idx_header[M] = header; // 걍 전체 header (나중에 export할때 써야함.)

			if (header.find("HLA") == 0) { // "HLA"로 시작하는지 확인
				col_idx_HLA[M] = header;
			}
			else {
				col_idx_SNP[M] = header;
			}

		}
	} else {
		std::cerr << "Error: File is empty or missing header line." << std::endl;
		exit(1);
	}


	// [0-3] 나머지 줄: 행렬 데이터 읽기
	matrix = arma::mat(M, M);  // Armadillo 행렬 초기화

	for (int row = 0; std::getline(file, line) && row < M; ++row) {

		std::istringstream iss(line);

		for (int col = 0; col < M; ++col) {

			double value;

			if (iss >> value) {
				matrix(row, col) = value;
			} else {
				std::cerr << "Error: Insufficient data in row " << row << std::endl;
				exit(1);
			}
		}
	}


	// check with printing
	matrix.print("LD matrix: ");

	std::cout << "col_inx_HLA contains " << col_idx_HLA.size() << " items:" << std::endl;

	for (const auto& pair : col_idx_HLA) {
		std::cout << "Key: " << pair.first << ", Value: " << pair.second << std::endl;
	}
	for (const auto& pair : col_idx_SNP) {
		std::cout << "Key: " << pair.first << ", Value: " << pair.second << std::endl;
	}



	// [1] SNPs들의 matrix로 subset하기

	// [1-1] subset하기.

	arma::uvec indices_to_keep;
	for (const auto& [key, index] : col_idx_SNP) {
		indices_to_keep.insert_rows(indices_to_keep.n_rows, arma::uvec({(arma::uword)key}));
	}

	inv_matrix_SNPs = matrix(indices_to_keep, indices_to_keep);

	inv_matrix_SNPs.print("The subsetted matrix_SNPs:");


	// [1-2] eigenvalues계산해놓기 => Log-likelihood계산할때 사용.
	eigenvalues_SNPs = arma::eig_sym(inv_matrix_SNPs);
	check_eigenvalues_positive(eigenvalues_SNPs);



	// [1-3] inverse해놓기. => Log-likelihood계산할때 사용.
	bool f_inv_success = arma::inv(inv_matrix_SNPs, inv_matrix_SNPs);

	if (!f_inv_success) {
		throw std::runtime_error("failed to compute inverse matrix!");
	}

	// check with printing
	eigenvalues_SNPs.print("eigenvalues_SNPs:");
	inv_matrix_SNPs.print("The inverted matrix_SNPs:");



}


void LDMatrix::check_eigenvalues_positive(const arma::vec& eigenvalues) {

	// Check if all eigenvalues are positive
	bool f_all_positive = std::all_of(eigenvalues.begin(), eigenvalues.end(), [](double val) {
		return val > 0;
	});

	// Throw an error if not all eigenvalues are positive
	if (!f_all_positive) {
		throw std::runtime_error("All eigenvalues of the LD correlation matrix must be positive! (Positive semi-definite)");
	}
}

// 얘 나중에 다시 가져와야 함.
// arma::mat subset_matrix(arma::mat& _mat, std::map<int, std::string>& col_idx_SNP) {
//
// 	arma::uvec indices_to_keep;
// 	for (const auto& [key, index] : col_idx_SNP) {
//
// 		indices_to_keep.insert_rows(indices_to_keep.n_rows, arma::uvec({(arma::uword)key}));
// 	}
//
// 	// 행렬 subset
// 	return _mat(indices_to_keep, indices_to_keep);
// }
