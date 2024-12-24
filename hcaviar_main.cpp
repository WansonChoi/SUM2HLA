// This is the main function, practically.
// The main procedure to calculate posterior prob.
#include "hcaviar_main.h"

#include <iostream>
#include <string>
#include <iomanip>

#include <armadillo>
#include "LDMatrix.h"
#include "GWAS_Z_observed.h"
#include "hCaviarModel.h"
#include "Util.h"

using namespace std;



bool hcaviar_main(const string& _ldFile, const string& _zFile, const string& _outputFileName,
	int _totalCausalSNP, double _NCP, double _rho, bool _histFlag, double _gamma,
	int _batch_size)
{

	// [0] load data

	// (0-1) load LD matrix
	LDMatrix ldmatrix(_ldFile);

	if (ldmatrix.M < _totalCausalSNP) {
		throw runtime_error(
			"The # of causal SNPs ('-c') should be less than the total # of the given SNPs.");
	}



	// (0-2) GWAS Z-score matrix
	GWAS_Z_observed Z_obs(_zFile);
	Z_obs.sort_z_scores(ldmatrix.col_idx_SNP);



	// [1; Main iteration] calculate posterior with the hCaviarModel instance.

	// (1-1) class static (global) 변수들 초기화.
	hCaviarModel::postValues_global.resize(ldmatrix.M, 0.0);
	hCaviarModel::histValues_global.resize(_totalCausalSNP + 1, 0.0);


	// (1-2) generate a "hCaviarModel" instance (with those two input data.)
	hCaviarModel hcaviar_temp(
		ldmatrix, Z_obs,
		_outputFileName, 0, _NCP, _rho, _histFlag, _gamma
	);


	double totalLikeLihoodLOG = 0.0;

	for (int _N_causal = 1; _N_causal <= _totalCausalSNP; _N_causal++) {

		cout << "\n===== N_causal: " << _N_causal << endl;

		// resetting the temp instance
		hcaviar_temp.N_causal = _N_causal;
		hcaviar_temp.reset_postValues();

		totalLikeLihoodLOG = hCaviarModel::addlogSpace(
			totalLikeLihoodLOG, hcaviar_temp.iterate_configures_given_N_causal()
		);


		// export; hcaviar_temp.export(); (필요에 따라 결과물 파일로 export하기)

	}


	// (3) wrap-up.

	// (3-1) pcausalSet 구하기.
	vector<char> pcausalSet(ldmatrix.M, '0');
	vector<int> rank(ldmatrix.M);

	// 전체 SNPs들에 대한 normalizing constant (for `postValues_global`)
	double total_post = 0.0;
	for(int i = 0; i < ldmatrix.M; i++)
		total_post = hCaviarModel::addlogSpace(total_post, hCaviarModel::postValues_global[i]);

	vector<Util::data> items;
	for(int i = 0; i < ldmatrix.M; i++) {
		//printf("%d==>%e ",i, postValues[i]/total_likelihood);
		items.push_back(Util::data(exp(hCaviarModel::postValues_global[i] - total_post), i, 0));
	}

	// sorting하기.
	std::sort(items.begin(), items.end(), Util::by_number());
	for(int i = 0; i < ldmatrix.M; i++)
		rank[i] = items[i].index1;


	double rho = 0.0;
	int index = 0;
	do{
		rho += exp(hCaviarModel::postValues_global[rank[index]] - total_post);
		pcausalSet[rank[index]] = '1'; // 뭐던동 '1'로 찍히는 애들이 causal임.
		// printf("%s %d %e\n", ldmatrix.col_idx_header[rank[index]], rank[index], rho);
		cout << ldmatrix.col_idx_header[rank[index]] << " " << rank[index] << " " << rho << endl;
		index++;
	} while(rho < _rho);

	// check with print
	// cout << "pcausalSet" << endl;
	// for (auto item : pcausalSet) {
	// 	cout << item << endl;
	// }


	// (3-2) export set
	export2File(_outputFileName+".total_log-likelihood.txt", totalLikeLihoodLOG); //Output the total likelihood to the log File
	printSet2File(_outputFileName+"_set", pcausalSet, ldmatrix.col_idx_header);
	printPost2File(_outputFileName+"_post", hCaviarModel::postValues_global, ldmatrix.col_idx_header, total_post, totalLikeLihoodLOG);
	printHist2File(_outputFileName+"_hist", hCaviarModel::histValues_global, _totalCausalSNP, totalLikeLihoodLOG);

	cout << "totalLikeLihoodLOG: " << totalLikeLihoodLOG << endl;
	cout << "totalPost: " << total_post << endl;

	return true;
}



bool hcaviar_main_2(const string& _ldFile, const string& _zFile, const string& _outputFileName,
	int _totalCausalSNP, double _NCP, double _rho, bool _histFlag, double _gamma)
{
	// [0] load data

	// (0-1) load LD matrix
	LDMatrix ldmatrix(_ldFile);

	if (ldmatrix.M < _totalCausalSNP) {
		throw runtime_error(
			"The # of causal SNPs ('-c') should be less than the total # of the given SNPs.");
	}



	// (0-2) GWAS Z-score matrix
	GWAS_Z_observed Z_obs(_zFile);
	Z_obs.sort_z_scores(ldmatrix.col_idx_SNP);






	return true;

};



bool hcaviar_main_3(const string& _ldFile, const string& _zFile, const string& _outputFileName,
	int _totalCausalSNP, double _NCP, double _rho, bool _histFlag, double _gamma)
{
	// [0] load data

	// (0-1) load LD matrix
	LDMatrix ldmatrix(_ldFile);

	if (ldmatrix.M < _totalCausalSNP) {
		throw runtime_error(
			"The # of causal SNPs ('-c') should be less than the total # of the given SNPs.");
	}

	// (0-2) GWAS Z-score matrix
	GWAS_Z_observed Z_obs(_zFile);
	Z_obs.sort_z_scores(ldmatrix.col_idx_SNP);



	// [1; Main iteration] calculate posterior with the hCaviarModel instance.

	// (1-1) class static (global) 변수들 초기화. (이 함수에서는 엄밀히 의미 없음)

	// hCaviarModel::postValues_global.resize(ldmatrix.M, 0.0);
	// hCaviarModel::histValues_global.resize(_totalCausalSNP + 1, 0.0);


	// (1-2) generate a "hCaviarModel" instance (with those two input data.)
	hCaviarModel hcaviar_temp(
		ldmatrix, Z_obs,
		_outputFileName, 0, _NCP, _rho, _histFlag, _gamma
	);


	/*
	- fwrite module은 a `N_causal` value에 대해서만 작동하도록 의도함.
		- (cf) <=> `hcaviar_main()` => for (int _N_causal = 1; _N_causal <= _totalCausalSNP; ++_N_causal)

	 */

	cout << "N_causal: " << _totalCausalSNP << " (only)" << endl;

	// resetting the temp instance
	hcaviar_temp.N_causal = _totalCausalSNP;
	hcaviar_temp.reset_postValues();

	hcaviar_temp.fwrite_logLikelihood(_outputFileName + ".c" + to_string(hcaviar_temp.N_causal));



	return 1;
}