#ifndef LDMATRIX_H
#define LDMATRIX_H

#include <iostream>
#include <fstream>
#include <string>
#include <map>

#include <armadillo>
#include <sstream>


struct LDMatrix {

	// member variables (public by default)
	int M = -1; // total number of the markers.
	std::map<int, std::string> col_idx_header; // 전부 담는 header

	std::map<int, std::string> col_idx_SNP;
	std::map<int, std::string> col_idx_HLA;
	std::map<int, std::string> col_idx_AA;
	std::map<int, std::string> col_idx_intraSNP;

	arma::mat matrix; // LD matrix 데이터 (SNP + HLA)

	arma::mat inv_matrix_SNPs; // subset한거 우선 여기 받고, eigenvaluesr구하고, 여기에 다시 inverse쳐놓을거임..
	arma::vec eigenvalues_SNPs;

	// member functions
	explicit LDMatrix(const std::string& filename);
	void print_markers(const std::string& _type);

	static void check_eigenvalues_positive(const arma::vec& eigenvalues);
};



#endif
