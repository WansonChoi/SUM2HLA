#include <iostream>
#include <fstream>
#include <string>

#include <armadillo>
#include "hCaviarModel.h"
#include "Util.h"
using namespace std;

std::vector<double> hCaviarModel::postValues_global;
std::vector<double> hCaviarModel::histValues_global;

hCaviarModel::hCaviarModel(
	const LDMatrix& _ldmat, const GWAS_Z_observed& _z_obs,
	const string& _outputFileName, int _N_causal, double _NCP, double _rho, bool _histFlag, double _gamma=0.01):
	ld_matrix(_ldmat), z_observed(_z_obs)
{

	N_causal = _N_causal;
	NCP = _NCP;
	gamma = _gamma;
	rho = _rho;
	histFlag = _histFlag;

	outputFileName = _outputFileName;


	snpCount = ld_matrix.M;
	// sigma      = nullptr; // deprecated
	// stat       = nullptr; // deprecated
	// pcausalSet = new char[snpCount]; // deprecated; moved to upper level
	// rank = new int[snpCount]; // deprecated; moved to upper level
	// snpNames   = new string [snpCount]; // deprecated


	// 'PostCal.h'에서 쓰던거
	postValues.resize(snpCount, 0.0);

	// log-likelihood계산 관련된 변수들 초기화.
	a_configure_uvec_temp = arma::uvec(ld_matrix.M);
	ncp_vector_temp = arma::vec(ld_matrix.M);

	indices_SNPs = arma::uvec(ld_matrix.col_idx_SNP.size()); // 얘한번만 구해놓으면 계속 쓸 수 있음.
	size_t idx = 0;
	for (const auto& [key, value] : ld_matrix.col_idx_SNP) {
		indices_SNPs(idx++) = key;
	}

	ncp_vector_SNPs_temp = arma::vec(ld_matrix.col_idx_SNP.size());
	diff = arma::vec(ld_matrix.col_idx_SNP.size());

	term1 = -0.5 * ld_matrix.M * std::log(2 * arma::datum::pi);
	term2 = -0.5 * arma::sum(arma::log(ld_matrix.eigenvalues_SNPs));

}

hCaviarModel::~hCaviarModel() {
	// delete[] pcausalSet;
	// delete[] rank;
}



void hCaviarModel::reset_postValues() {

	// (cf) 저 const는 객체 member 변수들을 건드리지 않겠다는 뜻임.
	// `postValues`와 `histValues`는 포인터 변수이기 때문에, 이 포인터 변수들의 주소를 바꾸지 않겠다는 뜻. 주소가 가리키는 값은 바꿔도 됨.

	for (int i = 0; i < snpCount; i++) {
		postValues[i] = 0.0;
	}

}



double hCaviarModel::iterate_configures_given_N_causal() {

	double sumLikelihood_N_causal = 0.0;

	const unsigned long long N_configures = comb(snpCount, N_causal);
	cout << "N_configures: " << N_configures << endl;

	vector<bool> a_configure;
	a_configure.resize(snpCount, 0);

	// The 1st configure (with initialization)
	std::fill(a_configure.begin(), a_configure.begin() + N_causal, 1);

	for (unsigned long long i_configure = 0; i_configure < N_configures; ++i_configure) {

		if (i_configure % 1000000 == 0) {
			cout << "===[" << i_configure << "] (" << current_time() << ")\n";

			for (int bit : a_configure) {
				std::cout << bit << " ";
			} std::cout << std::endl;

		}

		// (***) Log-likelihood calculation with the configure.
		double tmp_likelihood = calc_logLieklihood_of_the_configure(a_configure) + N_causal * log(gamma) + (snpCount - N_causal) * log(1-gamma);

		// cout << "Log-likelihood: " << tmp_likelihood << endl;
		// printf("%lf\n", tmp_likelihood);

		sumLikelihood_N_causal = addlogSpace(sumLikelihood_N_causal, tmp_likelihood);

		// 갈무리 (1) - postValues
		for (int j = 0; j < a_configure.size(); ++j) {
			postValues[j] = addlogSpace(postValues[j], tmp_likelihood * a_configure[j]);
		}

		prev_permutation(a_configure.begin(), a_configure.end()); // next configure.

	}

	// 갈무리 (2-1; global) - `histValues_global`
	histValues_global[N_causal] = addlogSpace(histValues_global[N_causal], sumLikelihood_N_causal);

	// 갈무리 (2-2; global) - `postValues_global`
	for (int j = 0; j < a_configure.size(); ++j) {
		// configurs들에 iterate하면서 누적한 `postValues[j]`를 한번만 누적하면 됨.
		postValues_global[j] = addlogSpace(postValues_global[j], postValues[j]);
	}

	return sumLikelihood_N_causal;
}



double hCaviarModel::calc_logLieklihood_of_the_configure(const vector<bool>& _a_configure) {

	// === [1] ncp mean vector 계산
	set_ncp_mean_vector(_a_configure); // `ncp_vector_temp`에 inplace로 값 채워놓음.
	// ncp_vector_temp.print("NCP mean vector:");

	ncp_vector_SNPs_temp = ncp_vector_temp(indices_SNPs);
	// ncp_vector_SNPs_temp.print("Subset NCP mean vector:");


	// === [4] log-likelihood

	// diff = z_scores - mean_vector
	diff = z_observed.z_scores_sorted_vec - ncp_vector_SNPs_temp;

	// Term 3: -0.5 * (z_scores - mean_vector)^T @ inv(corr_matrix) @ (z_scores - mean_vector)
	double mahalanobis_dist = arma::as_scalar(diff.t() * ld_matrix.inv_matrix_SNPs * diff); // Mahalanobis distance
	double term3 = -0.5 * mahalanobis_dist;

	// cout << "manhal: " << term3 << endl;

	// Total log-likelihood
	return term1 + term2 + term3;

}



void hCaviarModel::set_ncp_mean_vector(const vector<bool>& _a_configure) {

	// `a_configure_uvec_temp`와 `ncp_vector_temp`의 재활용. (새로 만들지 말고 최대한 재활용)

	// (1) `a_configure`를 Armadillo 벡터로 변환
	for (size_t i = 0; i < _a_configure.size(); ++i) {
		a_configure_uvec_temp(i) = _a_configure[i] ? 1 : 0; // bool을 1 또는 0으로 변환
	}

	// (2) Dot product: (13x13) * (13x1) -> (13x1)
	ncp_vector_temp = NCP * (ld_matrix.matrix * arma::conv_to<arma::vec>::from(a_configure_uvec_temp));

	// (3) `a_configure`에서 1인 위치는 값을 `_ncp`로 설정
	for (size_t i = 0; i < _a_configure.size(); ++i) {
		if (_a_configure[i]) {
			ncp_vector_temp(i) = NCP;
		}
	}

}



void hCaviarModel::export_result_N_causal(const string & _out) {



}



double hCaviarModel::addlogSpace(const double a, const double b) {

	if (a == 0)
		return b;
	if (b == 0)
		return a;

	double base = max(a,b);

	if (base - min(a,b) > 700)
		return base;

	return(base + log(1+exp(min(a,b)-base)));
}

// void finishUp() {
//
// 	// 얘네는 output을 보고 역으로 생각하면 좀 좋음.
//
// 	ofstream outputFile;
//     string outFileNameSet = string(outputFileName)+"_set";
//
//     outputFile.open(outFileNameSet.c_str());
//
//     for(int i = 0; i < snpCount; i++) {
//         if(pcausalSet[i] == '1')
//             outputFile << snpNames[i] << endl;
//     }
//
//     post->printPost2File(string(outputFileName)+"_post");
//
//     //output the histogram data to file
//     if(histFlag) // 이게 결국 '-f' flag. 뭔 histogram이라는데.
//         post->printHist2File(string(outputFileName)+"_hist");
// }
//
// // void printLogData() {
// // 	//print likelihood
// // 	//print all possible configuration from the p-causal set
// // 	post->computeALLCausalSetConfiguration(stat, NCP, pcausalSet,outputFileName+".log");
// // }


// void printHist2File(string fileName) {
// 	exportVector2File(fileName, histValues, maxCausalSNP+1);
// }
//
//
// void printPost2File(string fileName) {
//
// 	double total_post = 0;
//
// 	ofstream outfile(fileName.c_str(), ios::out );
//
// 	for(int i = 0; i < snpCount; i++)
// 		total_post = addlogSpace(total_post, postValues[i]);
//
// 	outfile << "SNP_ID\tProb_in_pCausalSet\tCausal_Post._Prob." << endl;
// 	for(int i = 0; i < snpCount; i++) {
// 		outfile << snpNames[i] << "\t" << exp(postValues[i]-total_post) << "\t" << exp(postValues[i]-totalLikeLihoodLOG) << endl;
// 	}
// }
