#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <map>
#include <set>

#include "GWAS_Z_observed.h"
#include <armadillo>

using namespace std;

GWAS_Z_observed::GWAS_Z_observed(const std::string& filename){ // instance만들면 바로 load.

	std::ifstream file(filename);

	if (!file.is_open()) {
		std::cerr << "Error: Unable to open file " << filename << std::endl;
		exit(1);
	}

	std::string line, header, snp_label;
	double z_score;

	// 첫 번째 줄: 헤더 확인
	if (!std::getline(file, line)) {
		std::cerr << "Error: File is empty or missing header line." << std::endl;
		exit(1);
	}

	// 이후 줄: SNP와 z-score 저장
	for (M = 0; std::getline(file, line); ++M) {

		std::istringstream iss(line);

		if (!(iss >> snp_label >> z_score)) {
			std::cerr << "Error: Invalid line format: " << line << std::endl;
			exit(1);
		}

		z_scores[snp_label] = z_score; // SNP-z-score 매핑 저장
	}

};


// LD matrix의 col_idx_SNP 순서에 따라 z-scores 정렬
// 기본적으론 사용자가 알아서 할거라고 가정.
void GWAS_Z_observed::sort_z_scores(const std::map<int, std::string>& col_idx_SNP) {

	// (a) SNP set 비교
	std::set<std::string> observed_snps, ld_snps;

	for (const auto& [snp_label, z_score] : z_scores) {
		observed_snps.insert(snp_label);
	}

	for (const auto& [idx, snp_label] : col_idx_SNP) {
		ld_snps.insert(snp_label);
	}

	if (observed_snps != ld_snps) {
		throw std::runtime_error("Error: SNP sets in observed z-scores and LD matrix do not match.");
	}


	z_scores_sorted.reserve(col_idx_SNP.size());

	// (b) LD matrix의 col_idx_SNP 순서에 따라 정렬
	for (const auto& [idx, snp_label] : col_idx_SNP) {
		z_scores_sorted.push_back(z_scores[snp_label]);
	}

	z_scores_sorted_vec = arma::vec(z_scores_sorted); // arma vector도 같이 준비.

	// check with printing
	z_scores_sorted_vec.print("Z-scores obs. sorted:");

}


// // [0-2] Load the GWAS z-score with a header.
// if (!z_observed.load(zFile)) {
// 	std::cerr << "Failed to load observed z-scores." << std::endl;
// }
//
// try {
// 	// z-scores 정렬 (LD matrix의 SNP set과 match하는 과정)
// 	z_observed.sort_z_scores(ld_matrix.col_idx_SNP);
//
// 	// 정렬된 z-scores 출력
// 	std::cout << "Sorted z-scores:" << std::endl;
// 	for (double z : z_observed.z_scores_sorted) {
// 		std::cout << z << std::endl;
// 	}
//
// } catch (const std::runtime_error& e) {
// 	std::cerr << e.what() << std::endl;
// }
