#include <iostream>
#include <fstream>
#include <string>

#include <armadillo>
#include "hCaviarModel.h"
#include "Util.h"
using namespace std;
using namespace arma;

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

	term1 = -0.5 * (ld_matrix.col_idx_SNP.size()) * std::log(2 * arma::datum::pi);
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



void hCaviarModel::fwrite_logLikelihood(const string & _out) {

	/*
	- prior 계산 안하고 log-likelihood만 계산.
	- 바로 fwrite만 수행.

	*/

	// (0-1) open file pointers.
	FILE* f_configure = fopen((_out + ".configures").c_str(), "w");
	FILE* f_logLL = fopen((_out + ".LL").c_str(), "w");

	if (!f_configure || !f_logLL) {
		std::cerr << "Error: Unable to open output files." << std::endl;
		if (f_configure) fclose(f_configure);
		if (f_logLL) fclose(f_logLL);
		exit(1);
	}

	// (0-2) set the initial configure.
	vector<bool> a_configure;
	a_configure.resize(snpCount, 0);

	std::fill(a_configure.begin(), a_configure.begin() + N_causal, 1);


	// (0-3) malloc a buffer.
	char* configure_buffer = (char*)malloc((snpCount + 1) * sizeof(char));
	if (configure_buffer == nullptr) {
		std::cerr << "Error: Memory allocation failed for configure_buffer." << std::endl;
		exit(1);
	}

	configure_buffer[snpCount] = '\n';      // 마지막에 줄바꿈 추가

	char log_buffer[64]; // Log-likelihood 기록용 버퍼


	// (1) the Main iteration

	const unsigned long long N_configures = comb(snpCount, N_causal);
	cout << "N_configures: " << N_configures << endl;

	for (unsigned long long i_configure = 0; i_configure < N_configures; ++i_configure) {

		if (i_configure % 1000000 == 0) {
			cout << "===[" << i_configure << "] (" << current_time() << ")\n";

			for (int bit : a_configure) {
				std::cout << bit << " ";
			} std::cout << std::endl;

		}

		double tmp_likelihood = calc_logLieklihood_of_the_configure(a_configure);

		// Configure를 buffer에 기록
		for (size_t i = 0; i < snpCount; ++i) {
			configure_buffer[i] = a_configure[i] ? '1' : '0';
		}
		fwrite(configure_buffer, sizeof(char), snpCount + 1, f_configure);

		// Log-likelihood 기록
		// fprintf(f_logLL, "%.10e\n", tmp_likelihood);
		int log_length = snprintf(log_buffer, sizeof(log_buffer), "%.20e\n", tmp_likelihood);
		if (log_length < 0 || log_length >= sizeof(log_buffer)) {
			std::cerr << "Error: Buffer overflow while formatting tmp_likelihood." << std::endl;
			exit(1);
		}
		fwrite(log_buffer, sizeof(char), log_length, f_logLL);


		prev_permutation(a_configure.begin(), a_configure.end()); // next configure.

	}

	cout << "The following two files were written:" << endl;
	cout << "\t" << _out + ".configures" << endl;
	cout << "\t" << _out + ".LL" << endl;


	free(configure_buffer);
	fclose(f_configure);
	fclose(f_logLL);

}



// double hCaviarModel::computeTotalLikelihood(int _totalCausalSNP) {
//
// 	int num = 0;
// 	double sumLikelihood = 0;
// 	double tmp_likelihood = 0;
// 	long int total_iteration = 0;
//
// 	int maxCausalSNP = _totalCausalSNP;
//
// 	int* configure = (int *) malloc (snpCount * sizeof(int *)); // original data
//
// 	// 모든 causal configurations들의 경우의 수를 미리 계산하고 시작 => 이 수 만큼 iteration을 돌도록 짬.
// 	for(long int i = 0; i <= maxCausalSNP; i++) // (ex) choose(50, 0) + c(50, 1) + c(50, 2) = 1 + 50 + 1225 = 1276
//  		total_iteration = total_iteration + nCr(snpCount, i);
// 	cout << "Max Causal=" << maxCausalSNP << endl;
//
// 	for(long int i = 0; i < snpCount; i++)
//  		configure[i] = 0; // 모두 0으로 초기화하고 시작.
//
// 	// Main iterations
// 	for(long int i = 0; i < total_iteration; i++) {
//
//  		tmp_likelihood = fastLikelihood(configure, stat, NCP) + num * log(gamma) + (snpCount-num) * log(1-gamma);
// 	    sumLikelihood = addlogSpace(sumLikelihood, tmp_likelihood);
//
//
//  		for(int j = 0; j < snpCount; j++) {
// 	         postValues_global[j] = addlogSpace(postValues_global[j], tmp_likelihood * configure[j]);
//  		}
//  		histValues_global[num] = addlogSpace(histValues_global[num], tmp_likelihood);
//  		/*for (int j = 0; j < snpCount; j++) {
//  			if (configure[j] != 0)
//  				cout << j << ",";
//  		}
//  		cout << " " << tmp_likelihood << endl;*/
//  		num = nextBinary(configure, snpCount);
//  		//cout << i << " "  << exp(tmp_likelihood) << endl;
//  		if(i % 1000 == 0)
//  			cerr << "\r                                                                 \r" << (double) (i) / (double) total_iteration * 100.0 << "%";
// 		}
//
// 	for(int i = 0; i <= maxCausalSNP; i++)
//  		histValues_global[i] = exp(histValues_global[i]-sumLikelihood);
//  		// `sumLikelihood`으로 나누는건 결국 histValues, 즉 causal에 상관없이 누적된 posterior values들임.
//
//
// 	free(configure);
//
// 	return(sumLikelihood);
// }
//
//
//
// double hCaviarModel::fastLikelihood(int* configure, double* stat, double NCP) {
//
// 	// 여기서 한번 SNP set으로
//
// 	int causalCount = 0;
// 	vector <int> causalIndex;
// 	for(int i = 0; i < snpCount; i++) {
// 		causalCount += configure[i];
// 		if(configure[i] == 1)
// 			causalIndex.push_back(i);
// 	}
//
// 	if (causalCount == 0) { // 쓸모 없는 block. `maxVal` 값을 안씀, 이후에.
// 		int maxVal = 0;
// 		for(int i = 0; i < snpCount; i++) {
// 			if (maxVal < abs(stat[i]))
// 				maxVal = stat[i];
// 		}
// 	}
//
// 	// causal SNP에 해당하는 z-scores과 LD 값들만 subset해오는 부분.
// 	mat Rcc(causalCount, causalCount, fill::zeros); // causalCount x causalCount
// 	mat Zcc(causalCount, 1, fill::zeros);
// 	mat mean(causalCount, 1, fill::zeros);
// 	mat diagC(causalCount, causalCount, fill::zeros); // causalCount x causalCount
//
// 	for (int i = 0; i < causalCount; i++){
// 		for(int j = 0; j < causalCount; j++) {
// 			Rcc(i,j) = sigmaMatrix(causalIndex[i], causalIndex[j]);
// 		}
// 		Zcc(i,0) = stat[causalIndex[i]];
// 		diagC(i,i) = NCP;
//
// 		// mean은 zeros로 fill하고 여기서 수정을 안함. => 아래 `fracdmvnorm`()함수에 그냥 0-vector로 들어감.
// 	}
//
// 	return fracdmvnorm(Zcc, mean, Rcc, diagC, NCP);
// }
//
//
// // We compute dmvnorm(Zcc, mean=rep(0,nrow(Rcc)), Rcc + Rcc %*% Rcc) / dmvnorm(Zcc, rep(0, nrow(Rcc)), Rcc))
// // togheter to avoid numerical over flow
// double hCaviarModel::fracdmvnorm(mat Z, mat mean, mat R, mat diagC, double NCP) {
//
// 	mat newR = R + R * diagC  * R; // diagC는 대각선이 ncp값으로 채워진 matrix.
// 	mat ZcenterMean = Z - mean; // 여기서 mean은 걍 0-vector임. `Z` 고대로 쓰는셈임.
//
// 	//mat res1 = trans(ZcenterMean) * inv(R) * (ZcenterMean);
// 	//mat res2 = trans(ZcenterMean) * inv(newR) *  (ZcenterMean);
// 	mat res1 = trans(ZcenterMean) * solve(R, eye(arma::size(R))) * (ZcenterMean);
// 	mat res2 = trans(ZcenterMean) * solve(newR, eye(arma::size(newR))) *  (ZcenterMean);
//
// 	double v1 = res1(0,0)/2-res2(0,0)/2;
// 	//CHANGE: MOVE FORM NORMAL CALCULATION TO LOG SPACE
// 	//return(exp(v1)/sqrt(det(newR))* sqrt(det(R)));
// 	return(v1 - log( sqrt(det(newR)) ) + log( sqrt(det(R)) ) );
// }
//
//
//
// int hCaviarModel::nextBinary(int* data, int size) {
//
// 	int i = 0;
// 	int total_one = 0;
// 	int index = size-1;
// 	int one_countinus_in_end = 0;
//
// 	while(index >= 0 && data[index] == 1) {
// 		index = index - 1;
// 		one_countinus_in_end = one_countinus_in_end + 1;
// 	}
// 	if(index >= 0) {
// 		while(index >= 0 && data[index] == 0) {
// 			index = index - 1;
// 		}
// 	}
// 	if(index == -1) {
// 		while(i <  one_countinus_in_end+1 && i < size) {
// 			data[i] = 1;
// 			i=i+1;
// 		}
// 		i = 0;
// 		while(i < size-one_countinus_in_end-1) {
// 			data[i+one_countinus_in_end+1] = 0;
// 			i=i+1;
// 		}
// 	}
// 	else if(one_countinus_in_end == 0) {
// 		data[index] = 0;
// 		data[index+1] = 1;
// 	} else {
// 		data[index] = 0;
// 		while(i < one_countinus_in_end + 1) {
// 			data[i+index+1] = 1;
// 			if(i+index+1 >= size)
// 				printf("ERROR3 %d\n", i+index+1);
// 			i=i+1;
// 		}
// 		i = 0;
// 		while(i < size - index - one_countinus_in_end - 2) {
// 			data[i+index+one_countinus_in_end+2] = 0;
// 			if(i+index+one_countinus_in_end+2 >= size) {
// 				printf("ERROR4 %d\n", i+index+one_countinus_in_end+2);
// 			}
// 			i=i+1;
// 		}
// 	}
// 	i = 0;
// 	total_one = 0;
// 	for(i = 0; i < size; i++)
// 		if(data[i] == 1)
// 			total_one = total_one + 1;
//
// 	return(total_one);
// }



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
