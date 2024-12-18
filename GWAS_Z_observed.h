#ifndef GWAS_Z_OBSERVED_H
#define GWAS_Z_OBSERVED_H

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <map>
#include <set>

#include <armadillo>


struct GWAS_Z_observed {

	int M = -1;

	// Observed z-scores 저장
	std::unordered_map<std::string, double> z_scores;
	std::vector<double> z_scores_sorted{};
	arma::vec z_scores_sorted_vec{};

	explicit GWAS_Z_observed(const std::string& filename);
	void sort_z_scores(const std::map<int, std::string>& col_idx_SNP);

};


#endif
