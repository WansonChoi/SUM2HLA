#ifndef HCAVIARMODEL_H
#define HCAVIARMODEL_H

#include <iostream>
#include <fstream>
#include <string>


#include <armadillo>

#include "LDMatrix.h"
#include "GWAS_Z_observed.h"

// using namespace arma;


/*
- 'PostCal' class와 여기 'CaviarModel' class를 merge했음, 많은 내용이 너무 중복되어서.

- each N_causal에 대해 log-likelihood 계산 & 누적을 하도록 class를 정의하는게 낫지 않나 싶음.
	- <=> N_causal = [1, 2, 3] => comb(M, 1) + comb(M, 2) + comb(M, 3)
	- Each comb(M, _)에 하나씩 돌릴 수 있도록, 대응되도록.
	- 그리고 이 결과를 export할 수 있도록.
		결국 export만 잘 해놓으면 누적해서 posterior계산 가능 함.
	- 또, 여차하면 예전에 저렇게도 돌릴 수도 있도록.

- 확실한건, 비록 FP는 아닐지라도, FP하듯이 atomic한 단위를 define학고 얘에 대해 iteration건다 생각하면 편함.
	- a sequence of transformations.
 */



struct hCaviarModel{

	int N_causal;
	double NCP;
	double gamma;
	double rho;
    bool histFlag;

	// string * snpNames;
	// string ldFile;
	//    string zFile;
	std::string outputFileName;
	// string geneMapFile;

	int snpCount;
	// double * sigma;
 //    double * stat;
    // char* pcausalSet{};
    // int* rank{};


	const LDMatrix& ld_matrix;
	const GWAS_Z_observed& z_observed;


	// 'PostCal.h'에서 쓰던거
	static std::vector<double> postValues_global;	//the posterior value for each SNP being causal
	static std::vector<double> histValues_global;	//the probability of the number of causal SNPs, we make the histogram of the causal SNPs

	std::vector<double> postValues;
	// std::vector<double> histValues;


	/*
	- For each `N_causal` value 마다 calculation을 하도록 바꿨음, where N_causal = [1, 2, 3, ...].
	- 문제는 each `N_causal` value 마다 hCaviarModel의 instance를 만들면 시간, 메모리 둘 다 손해임.
		- (맨 처음에는 그냥 the hCaviarModel instance를 여러 개 만들려 그랬음.)
	- 그래서 instance는 하나만 만들고, 얘를 temporary instance로 여러번 쓰기로 함.
	*/

	// 재활용할 temp variables들, log-likelihood 계산할 때.
	// ncp mean vector의 length가 최대 50k까지 갈 수 있음. 얘를 매번 generate 하는 상황을 막기 위함임.
	arma::uvec a_configure_uvec_temp;
	arma::vec ncp_vector_temp;

	arma::uvec indices_SNPs;

	arma::vec ncp_vector_SNPs_temp;
	arma::vec diff; // z_obs - mean;

	double term1;
	double term2;



	hCaviarModel(
		const LDMatrix& _ldmat, const GWAS_Z_observed& _z_obs,
		const std::string& _outputFileName, int _N_causal, double _NCP, double _rho, bool _histFlag, double _gamma
	);
	~hCaviarModel();

	void reset_postValues();
	double iterate_configures_given_N_causal();
	double calc_logLieklihood_of_the_configure(const std::vector<bool>& _a_configure);

	void set_ncp_mean_vector(const std::vector<bool>& _a_configure);

	void fwrite_logLikelihood(const std::string&);

	// Static functions
	static double addlogSpace(double, double);


	// Plain Cavaiar
	double computeTotalLikelihood(int _totalCausalSNP);
	static int nextBinary(int* data, int size);
	double fastLikelihood(int* configure, double* stat, double NCP);
	double fracdmvnorm(arma::mat Z, arma::mat mean, arma::mat R, arma::mat diagC, double NCP);

	// void printHist2File(string);
	// void printPost2File(string);

	// // hCAVIAR by WC.
	// double computeTotalLikelihood_WC(double * stat, double NCP) ;
	// double findOptimalSetGreedy_WC(double * stat, double NCP, char * pcausalSet, int *rank,  double inputRho, string outputFileName);
	//
	// unsigned long long comb(int n, int k);
	// vector<vector<bool>> generate_configures_batch(int batch_size, vector<bool> v_start, bool f_discard_1st);
	// double fastLikelihood_WC(vector<bool> &a_configure, double NCP);



};
 
#endif
