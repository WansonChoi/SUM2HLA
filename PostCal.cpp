#include <vector>
#include <algorithm>
#include <set>
#include <iostream>
#include <armadillo>
#include <iomanip> 

#include "Util.h"
#include "PostCal.h"

using namespace arma;


void printGSLPrint(mat &A, int row, int col) {
	for(int i = 0; i < row; i++) {
		for(int j = 0; j < col; j++)
			printf("%g ", A(i, j));
		printf("\n");
	}	
}

// string PostCal::convertConfig2String(int * config, int size) {
// 	string result = "0";
// 	for(int i = 0; i < size; i++)
// 		if(config[i]==1)
// 			result+= "_" + convertInt(i);
// 	return result;
// }
//
// // We compute dmvnorm(Zcc, mean=rep(0,nrow(Rcc)), Rcc + Rcc %*% Rcc) / dmvnorm(Zcc, rep(0, nrow(Rcc)), Rcc))
// // togheter to avoid numerical over flow
// double PostCal::fracdmvnorm(mat Z, mat mean, mat R, mat diagC, double NCP) {
//
//     mat newR = R + R * diagC  * R; // diagC는 대각선이 ncp값으로 채워진 matrix.
//     mat ZcenterMean = Z - mean; // 여기서 mean은 걍 0-vector임. `Z` 고대로 쓰는셈임.
//
//     //mat res1 = trans(ZcenterMean) * inv(R) * (ZcenterMean);
//     //mat res2 = trans(ZcenterMean) * inv(newR) *  (ZcenterMean);
// 	mat res1 = trans(ZcenterMean) * solve(R, eye(size(R))) * (ZcenterMean);
// 	mat res2 = trans(ZcenterMean) * solve(newR, eye(size(newR))) *  (ZcenterMean);
//
// 	double v1 = res1(0,0)/2-res2(0,0)/2;
// 	//CHANGE: MOVE FORM NORMAL CALCULATION TO LOG SPACE
//     //return(exp(v1)/sqrt(det(newR))* sqrt(det(R)));
//     return(v1 - log( sqrt(det(newR)) ) + log( sqrt(det(R)) ) );
// }
//
// // We compute dmvnorm(Zcc, mean=rep(0,nrow(Rcc)), Rcc + Rcc %*% Rcc) / dmvnorm(Zcc, rep(0, nrow(Rcc)), Rcc))
// // togheter to avoid numerical over flow, We deal with singular LD matrix
// // eign decomposition matrx R
// // R = Q M Q^T where M is the diagonal matrix of eign values
// double PostCal::fracdmvnorm2(mat Z, mat mean, mat R, mat diagC, double NCP) {
// 	int rowCount = R.n_rows;
// 	double MDet=1;
// 	mat Q = zeros(rowCount, rowCount);
// 	mat eignVec;
// 	vec eignVal;
// 	mat MHalfInv = zeros(rowCount, rowCount);
// 	mat MHalf = zeros(rowCount, rowCount);
// 	eig_sym(eignVal, eignVec, R);
// 	mat ZcenterMean = Z - mean;
// 	uvec indices;
// 	indices = sort_index(abs(ZcenterMean));
// 	for(int i = 0 ; i < rowCount; i++){
// 		if(eignVal[indices[i]] > 0) {
// 			MHalfInv(i,i) = 1/sqrt(eignVal[indices[i]]);
// 			MDet = MDet * eignVal[indices[i]];
// 			MHalf(i,i) = sqrt(eignVal[indices[i]]);
// 			for (int j = 0; j < rowCount; j++){
// 				Q(i,j) = eignVec(indices[i],j);
// 			}
// 		}
// 	}
// 	mat ZcenterMeanTilda = MHalfInv * Q.t() * ZcenterMean;
// 	//mat res1 = ZcenterMeanTilda.t() * ZcenterMeanTilda;
// 	//double v1 = -res1(0,0)/2 - log (sqrt(MDet));
// 	mat MHalfQ = MHalf * Q.t();
// 	//mat res2 = ZcenterMeanTilda.t() * inv(eye(rowCount, rowCount) + MHalfQ * diagC * MHalfQ.t())  * ZcenterMeanTilda;
// 	mat res3 = ZcenterMeanTilda.t() * (inv(eye(rowCount, rowCount) + MHalfQ * diagC * MHalfQ.t())-eye(rowCount,rowCount)) * ZcenterMeanTilda;
// 	//double v2 = -res2(0,0)/2 - log (sqrt(MDet*det(eye(rowCount, rowCount)+diagC*R)));
// 	double v3 = -res3(0,0)/2 - log (sqrt(MDet*det(eye(rowCount, rowCount)+diagC*R))) + log (sqrt(MDet));
// 	return (v3);
// }
//
// double PostCal::dmvnorm(mat Z, mat mean, mat R) {
//         mat ZcenterMean = Z - mean;
//         mat res = trans(ZcenterMean) * inv(R) * (ZcenterMean);
//         double v1 = res(0,0);
//         double v2 = log(sqrt(det(R)));
//         return (exp(-v1/2-v2));
// }

// cc=causal SNPs
// Rcc = LD of causal SNPs
// Zcc = Z-score of causal SNPs
// dmvnorm(Zcc, mean=rep(0,nrow(Rcc)), Rcc + Rcc %*% Rcc) / dmvnorm(Zcc, rep(0, nrow(Rcc)), Rcc))
//

// double PostCal::fastLikelihood(int * configure, double * stat, double NCP) {
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

/*
 * This function is not depricated and invSigmaMatrix is not used anymore
 */
// double PostCal::likelihood(int * configure, double * stat, double NCP) {
// 	int causalCount = 0;
// 	int index_C = 0;
//         double matDet = 0;
// 	double res    = 0;
//
// 	for(int i = 0; i < snpCount; i++)
// 		causalCount += configure[i];
// 	if(causalCount == 0){
// 		mat tmpResultMatrix1N = statMatrixtTran * invSigmaMatrix;
// 		mat tmpResultMatrix11 = tmpResultMatrix1N * statMatrix;
// 		res = tmpResultMatrix11(0,0);
// 		matDet = sigmaDet;
// 		return( exp(-res/2)/sqrt(abs(matDet)) );
// 	}
// 	mat U(snpCount, causalCount, fill::zeros);
// 	mat V(causalCount, snpCount, fill::zeros);
// 	mat VU(causalCount, causalCount, fill::zeros);
//
// 	for(int i = 0; i < snpCount; i++) {
//                 if (configure[i] == 0)	continue;
//                 else {
//                         for(int j = 0; j < snpCount; j++)
//                                 U(j, index_C) = sigmaMatrix(j,i);
// 			V(index_C, i) = NCP;
//                         index_C++;
//                 }
//         }
// 	VU = V * U;
// 	mat I_AA   = mat(snpCount, snpCount, fill::eye);
// 	mat tmp_CC = mat(causalCount, causalCount, fill::eye)+ VU;
// 	matDet = det(tmp_CC) * sigmaDet;
// 	mat tmp_AA = invSigmaMatrix - (invSigmaMatrix * U) * pinv(tmp_CC) * V ;
// 	//tmp_AA     = invSigmaMatrix * tmp_AA;
// 	mat tmpResultMatrix1N = statMatrixtTran * tmp_AA;
//         mat tmpResultMatrix11 = tmpResultMatrix1N * statMatrix;
//         res = tmpResultMatrix11(0,0);
//
// 	if(matDet==0) {
// 		cout << "Error the matrix is singular and we fail to fix it." << endl;
// 		exit(0);
// 	}
// 	/*
// 		We compute the log of -res/2-log(det) to see if it is too big or not.
// 		In the case it is too big we just make it a MAX value.
// 	*/
// 	double tmplogDet = log(sqrt(abs(matDet)));
// 	double tmpFinalRes = -res/2 - tmplogDet;
// 	if(tmpFinalRes > 700)
// 		return(exp(700));
// 	return( exp(-res/2)/sqrt(abs(matDet)) );
// }

// int PostCal::nextBinary(int * data, int size) {
// 	int i = 0;
// 	int total_one = 0;
// 	int index = size-1;
//     int one_countinus_in_end = 0;
//
//         while(index >= 0 && data[index] == 1) {
//                 index = index - 1;
//                 one_countinus_in_end = one_countinus_in_end + 1;
// 	}
// 	if(index >= 0) {
//         	while(index >= 0 && data[index] == 0) {
//                	 index = index - 1;
// 		}
// 	}
//         if(index == -1) {
//                 while(i <  one_countinus_in_end+1 && i < size) {
//                         data[i] = 1;
//                         i=i+1;
// 		}
//                 i = 0;
//                 while(i < size-one_countinus_in_end-1) {
//                         data[i+one_countinus_in_end+1] = 0;
//                         i=i+1;
// 		}
// 	}
//         else if(one_countinus_in_end == 0) {
//                 data[index] = 0;
//                 data[index+1] = 1;
// 	} else {
//                 data[index] = 0;
//                 while(i < one_countinus_in_end + 1) {
//                         data[i+index+1] = 1;
// 			if(i+index+1 >= size)
// 				printf("ERROR3 %d\n", i+index+1);
//                         i=i+1;
// 		}
//                 i = 0;
//                 while(i < size - index - one_countinus_in_end - 2) {
//                         data[i+index+one_countinus_in_end+2] = 0;
// 			if(i+index+one_countinus_in_end+2 >= size) {
// 				printf("ERROR4 %d\n", i+index+one_countinus_in_end+2);
// 			}
//                         i=i+1;
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
//
// double PostCal::computeTotalLikelihood(double * stat, double NCP) {
//
// 	int num = 0;
// 	double sumLikelihood = 0;
// 	double tmp_likelihood = 0;
// 	long int total_iteration = 0 ;
// 	int * configure = (int *) malloc (snpCount * sizeof(int *)); // original data
//
// 	// 모든 causal configurations들의 경우의 수를 미리 계산하고 시작 => 이 수 만큼 iteration을 돌도록 짬.
// 	for(long int i = 0; i <= maxCausalSNP; i++) // (ex) choose(50, 0) + c(50, 1) + c(50, 2) = 1 + 50 + 1225 = 1276
// 		total_iteration = total_iteration + nCr(snpCount, i);
// 	cout << "Max Causal=" << maxCausalSNP << endl;
//
// 	for(long int i = 0; i < snpCount; i++)
// 		configure[i] = 0; // 모두 0으로 초기화하고 시작.
//
// 	// Main iterations
// 	for(long int i = 0; i < total_iteration; i++) {
//
// 		tmp_likelihood = fastLikelihood(configure, stat, NCP) + num * log(gamma) + (snpCount-num) * log(1-gamma);
//         sumLikelihood = addlogSpace(sumLikelihood, tmp_likelihood);
//
// 		// tmp_likelihood 얘 왠지 equation (2)에서 분자 부분에 log 친 결과같지 않냐?
// 		// P(s_hat | c) * P(c) = f(lambda_c, 0, Sigma_c) * Product(0 <= i < M; gamma^(c_i)*gamma^(1-c_i))
// 		// log(P(c)) => |c_i|*log(gamma) + |1-c_i|*log(1-gamma)
//
// 		// sumLikelihood는 뭐던동 분모, normalizing constants, 의 log값인가 싶네.
//
//
// 		for(int j = 0; j < snpCount; j++) {
//             postValues[j] = addlogSpace(postValues[j], tmp_likelihood * configure[j]);
// 		}
// 		histValues[num] = addlogSpace(histValues[num], tmp_likelihood);
// 		/*for (int j = 0; j < snpCount; j++) {
// 			if (configure[j] != 0)
// 				cout << j << ",";
// 		}
// 		cout << " " << tmp_likelihood << endl;*/
// 		num = nextBinary(configure, snpCount);
// 		//cout << i << " "  << exp(tmp_likelihood) << endl;
// 		if(i % 1000 == 0)
// 			cerr << "\r                                                                 \r" << (double) (i) / (double) total_iteration * 100.0 << "%";
// 	}
// 	for(int i = 0; i <= maxCausalSNP; i++)
// 		histValues[i] = exp(histValues[i]-sumLikelihood);
// 		// `sumLikelihood`으로 나누는건 결국 histValues, 즉 causal에 상관없이 누적된 posterior values들임.
//         free(configure);
//         return(sumLikelihood);
//
// 	// 결국 return하는건 `sumLikelihood`임, 모든 causal configurations들에 iterate해서.
// 	// 아무래도 얘가 normalizing constant인 것 같음, 분모부분.
// 	// 아 그러면 결국 `addlogSpace`는 log space에서의 덧셈을 수행하는 건가보네.
// }
//
// bool PostCal::validConfigutation(int * configure, char * pcausalSet) {
// 	for(int i = 0; i < snpCount; i++){
// 		if(configure[i] == 1 && pcausalSet[i] == '0')
// 			return false;
// 	}
// 	return true;
// }
//
// /*
//  * This is a auxilary function used to generate all possible causal set that
//  * are selected in the p-causal set
// */
// void PostCal::computeALLCausalSetConfiguration(double * stat, double NCP, char * pcausalSet, string outputFileName) {
// 	int num = 0;
//         double sumLikelihood = 0;
//         double tmp_likelihood = 0;
//         long int total_iteration = 0 ;
//         int * configure = (int *) malloc (snpCount * sizeof(int *)); // original data
//
//         for(long int i = 0; i <= maxCausalSNP; i++)
//                 total_iteration = total_iteration + nCr(snpCount, i);
//         for(long int i = 0; i < snpCount; i++)
//                 configure[i] = 0;
//         for(long int i = 0; i < total_iteration; i++) {
// 		if (validConfigutation(configure, pcausalSet)) {
// 			//log space
//                 	tmp_likelihood = fastLikelihood(configure, stat, NCP) +  num * log(gamma) + (snpCount-num) * log(1-gamma);
// 			exportVector2File(outputFileName, configure, snpCount);
// 			export2File(outputFileName, tmp_likelihood);
// 		}
// 		num = nextBinary(configure, snpCount);
// 	}
// }
//
// /*
// 	stat is the z-scpres
// 	sigma is the correaltion matrix
// 	G is the map between snp and the gene (snp, gene)
// */
// double PostCal::findOptimalSetGreedy(double * stat, double NCP, char * pcausalSet, int *rank,  double inputRho, string outputFileName) {
//
// 	int index = 0;
//     double rho = 0;
//     double total_post = 0;
//
// 	// [1] iter on each causal configuration (given # of causal SNPs).
//     totalLikeLihoodLOG = computeTotalLikelihood(stat, NCP);
//
// 	/*
// 	 * - `totalLikeLihoodLOG` => normalizing constant in equation 2 (i.e., summed denominators on the causal configurations.)
// 	 * - histValues[num] := the probability of the number of causal SNPs (얘를 histogram이라고 표현함)
// 	 * - a causal configuration이 주어졌을 때, equation (3)을 계산한 부분을 싹 다 더한 값.
// 	 */
//
// 	// 주어진 전체 configurations들 iteration하면서 each # of causl SNPs일때의 joint porb만 gathering함.
// 	// 결국 모든 configure의 likelihood를 # of causal SNPs values들로 partition한 꼴.
// 	// 이 값들을 모두 더해놓은 totalLikeLihoodLOG도 있으니 이걸로 나누면 the posterior probability of the number of causal SNPs 뚝딱.
//
// 	// 얘가 '-f' argument 주어지면 write하는 정보임.
// 	// 저 함수내에서 histValues들에 대해서 나누는 데만 (logspace에서 빼서) 씀. (이 외에는 쓰지 않음).
// 	// log-likelihood로 계산한거 같음. 아래에서도 likelihood를 exp로 계산해서 write함.
//
// 	export2File(outputFileName+".log", exp(totalLikeLihoodLOG)); //Output the total likelihood to the log File
//
// 	// [2] PIP 계산.
// 	for(int i = 0; i < snpCount; i++)
// 		total_post = addlogSpace(total_post, postValues[i]);
//
// 	/*
// 	 * 모든 SNP들에서의 `postValues`들을 싹 다 더함 => `total_post`
// 	 * Each SNP의 `postValues[i]`를 `total_post`로 나눔. => Each SNP의 Posterior prob.
// 	 */
//
// 	// 이게 PIP 인 것 같음.
// 		// 왜냐면, a causal configuration이 주어지고 equation (3)의 joint prob.가 주어졌을 때 (`tmp_likelihood`),
// 		// `tmp_likelihood * configure[j]` 형태로, 주어진 causal configuration에서 casual인 SNP한테만 이 값을 누적함.
// 	// PIP => a SNP이 causal일 확률.
// 		// max # of causal SNPs값이 주어졌을 때, 이때 고려할 수 있는 모든 configurations들이 주어지고,
// 		// a configuration의 joint prob.을 해당 configuration의 causal SNPs들 한테만 누적하면 됨.
//
// 	// `postValues`와 `histValues`들은 `computeTotalLikelihood()`함수 내에서 inplace로 채워져서 옴.
//
// 	printf("Total Likelihood= %e SNP=%d \n", total_post, snpCount);
//
//     std::vector<data> items;
//     std::set<int>::iterator it;
// 	//output the poster to files
//     for(int i = 0; i < snpCount; i++) {
//          //printf("%d==>%e ",i, postValues[i]/total_likelihood);
//          items.push_back(data(exp(postValues[i]-total_post), i, 0));
//     	// 빼고 exp위에 올리는게 결국 M(=50) SNPs들에 대한 normalizing constant로 나누는 과정인 듯.
//     }
//     printf("\n");
//
//
// 	// [3] credible set 구하기.
//     std::sort(items.begin(), items.end(), by_number());
//     for(int i = 0; i < snpCount; i++)
//         rank[i] = items[i].index1;
//
//     for(int i = 0; i < snpCount; i++)
//         pcausalSet[i] = '0';
//
//     do{
//         rho += exp(postValues[rank[index]]-total_post);
//         pcausalSet[rank[index]] = '1'; // 뭐던동 '1'로 찍히는 애들이 causal임.
//         printf("%d %e\n", rank[index], rho);
//         index++;
//     } while(rho < inputRho);
//
// 	/*
// 	 * 걍 PIP 값들로 sorting하고, 높은 순으로 PIP 값들 더했을 때 `inputRho`값 미만이 될만큼만 떼어 옴.
// 	 */
//
//     printf("\n");
//
// 	return(0);
//
// 	// 이론상 여기까지 오면 CAVIAR 끝임.
// }


// 여기서 부터 나중에 파일로 빼라.
// 지금은 시간없어서 일단 여기에 다 넣고 돌리기.

double PostCal::findOptimalSetGreedy_WC(double * stat, double NCP, char * pcausalSet, int *rank,  double inputRho, string outputFileName) {


	int index = 0;
	double rho = 0;
	double total_post = 0;

	// [1] iter on each causal configuration (given # of causal SNPs).
	totalLikeLihoodLOG = computeTotalLikelihood_WC(stat, NCP);

	export2File(outputFileName+".log", exp(totalLikeLihoodLOG)); //Output the total likelihood to the log File


	// [2] PIP 계산.
	for(int i = 0; i < snpCount; i++)
		total_post = addlogSpace(total_post, postValues[i]);

	printf("Total Likelihood= %e SNP=%d \n", total_post, snpCount);

	std::vector<data> items;
	std::set<int>::iterator it;
	//output the poster to files
	for(int i = 0; i < snpCount; i++) {
		//printf("%d==>%e ",i, postValues[i]/total_likelihood);
		items.push_back(data(exp(postValues[i]-total_post), i, 0));
		// 빼고 exp위에 올리는게 결국 M(=50) SNPs들에 대한 normalizing constant로 나누는 과정인 듯.
	}
	printf("\n");



	// [3] credible set 구하기.
	std::sort(items.begin(), items.end(), by_number());
	for(int i = 0; i < snpCount; i++)
		rank[i] = items[i].index1;

	for(int i = 0; i < snpCount; i++)
		pcausalSet[i] = '0';

	do{
		rho += exp(postValues[rank[index]]-total_post);
		pcausalSet[rank[index]] = '1'; // 뭐던동 '1'로 찍히는 애들이 causal임.
		printf("%d %e\n", rank[index], rho);
		index++;
	} while(rho < inputRho);

	printf("\n");

	return(0);
}



double PostCal::computeTotalLikelihood_WC(double * stat, double NCP) {

	// (1) Total likelihood, (2) postValue, (3) histValue 계산하기

	int batch_size = 32;

	double sumLikelihood = 0.0;


	// 차이가 있다면 N_causal로 iteration하나가 더 낌.
	for (int _N_causal = 1; _N_causal <= maxCausalSNP; _N_causal++) {

		cout << "\n===== N_causal: " << _N_causal << endl;

		unsigned long long _N_configures = comb(snpCount, _N_causal);
		int total_batches = (_N_configures + batch_size - 1) / batch_size; // 올림 처리 (걍 ceil function 구현한 거)

		cout << "_N_configures: " << _N_configures << endl;
		cout << "total_batches: " << total_batches << endl;


		// `configures_batch`의 init. (살짝 더러움.)
		std::vector<bool> temp (snpCount, 0);
		std::fill(temp.begin(), temp.begin() + _N_causal, 1);

		std::vector<std::vector<bool>> configures_batch;
		configures_batch.push_back(temp);

		// Note that `_N_configures` can be very large.
		// So, we introduced the batch, which is a chunk of `_N_configures`.
		for (int i_batch = 0; i_batch < total_batches; i_batch++) {

			cout << "i_batch: " << i_batch << endl;

			configures_batch =
				generate_configures_batch(batch_size, configures_batch.back(), (i_batch > 0));

			vector<double> log_likelihood_temp(configures_batch.size(), 0.0); // 어차피 batch size * 8 bytes 만 있으면 돼서 문제 없을 듯?

			// #pragma omp parallel for
			for (int i_configure_batch = 0; i_configure_batch < configures_batch.size(); i_configure_batch++) {
				cout << "===[" << i_batch*batch_size + i_configure_batch << "]" << endl;

				vector<bool> configure_temp = configures_batch[i_configure_batch];

				for (int bit : configure_temp) {
					std::cout << bit << " ";
				} std::cout << std::endl;


				double tmp_likelihood = fastLikelihood_WC(configure_temp, NCP) + _N_causal * log(gamma) + (snpCount - _N_causal) * log(1-gamma);

				log_likelihood_temp[i_configure_batch] = tmp_likelihood;
			}

			// 위 for문 분리해서 re-iterate.
			// 결과 취합은 굳이 thread까는게 overhead가 더 들 것 같았음.
			for (int i_configure_batch = 0; i_configure_batch < configures_batch.size(); i_configure_batch++) {

				double tmp_likelihood = log_likelihood_temp[i_configure_batch];
				vector<bool> configure_temp = configures_batch[i_configure_batch];

				// (1) totalLikelihood
				sumLikelihood = addlogSpace(sumLikelihood, tmp_likelihood);

				cout << "tmp_likelihood: " << tmp_likelihood << endl;
				cout << "sumLikelihood: " << sumLikelihood << endl;

				// (2) `postValues`
				for (int j = 0; j < snpCount; j++) {
					postValues[j] = addlogSpace(postValues[j], tmp_likelihood * configure_temp[j]);
				}

				// (3) 'histValues'
				histValues[_N_causal] = addlogSpace(histValues[_N_causal], sumLikelihood);

			}
		}
	}


	for(int i = 0; i < maxCausalSNP; i++) {
		histValues[i] = exp(histValues[i]-sumLikelihood);
	}

	return sumLikelihood;

}



unsigned long long PostCal::comb(int n, int k) {

	if (k > n) return 0;
	if (k == 0 || k == n) return 1;

	// Create a 1D DP array
	std::vector<unsigned long long> dp(k + 1, 0);
	dp[0] = 1; // comb(i, 0) = 1

	// Fill DP array using Pascal's Triangle property
	for (int i = 1; i <= n; ++i) {
		for (int j = std::min(i, k); j > 0; --j) {
			dp[j] += dp[j - 1];
		}
	}

	return dp[k];
}



vector<vector<bool>> PostCal::generate_configures_batch(int batch_size, vector<bool> v_start, bool f_discard_1st=false) {

	vector<vector<bool>> result;
	result.reserve(batch_size);

	// cout << "start vector:" << endl;
	// for (auto v : v_start) {
	// 	cout << v << " ";
	// }cout << endl;


	if (f_discard_1st) {
		// discard the 1st item, which is redundant from the 2nd batch.
		std::prev_permutation(v_start.begin(), v_start.end());
	}

	int i = 0;
	do {
		result.push_back(v_start);
		i++;
	} while (i < batch_size && prev_permutation(v_start.begin(), v_start.end()));

	return result;

}



arma::vec get_ncp_mean_vector(arma::mat& LD_mat, const std::vector<bool>& _a_configure, double _ncp = 5.2) {

	// (1) `a_configure`를 Armadillo 벡터로 변환
	arma::uvec a_configure_uvec(_a_configure.size());
	for (size_t i = 0; i < _a_configure.size(); ++i) {
		a_configure_uvec(i) = _a_configure[i] ? 1 : 0; // bool을 1 또는 0으로 변환
	}

	// (2) Dot product: (13x13) * (13x1) -> (13x1)
	arma::vec ncp_vector = _ncp * (LD_mat * arma::conv_to<arma::vec>::from(a_configure_uvec));

	// (3) `a_configure`에서 1인 위치는 값을 `_ncp`로 설정
	for (size_t i = 0; i < _a_configure.size(); ++i) {
		if (_a_configure[i]) {
			ncp_vector(i) = _ncp;
		}
	}

	return ncp_vector; // (13x1) 벡터 반환
}


// Log-likelihood 계산
double calc_log_likelihood_mvn(
	const arma::vec& obs_zscores,         // Observed z-scores
	const arma::vec& mean_vector,        // Mean vector
	const arma::mat& inv_corr_matrix,    // Inverse of correlation matrix
	double term1,                        // Precomputed Term 1
	double term2                         // Precomputed Term 2
) {
	// diff = z_scores - mean_vector
	arma::vec diff = obs_zscores - mean_vector;

	// Term 3: -0.5 * (z_scores - mean_vector)^T @ inv(corr_matrix) @ (z_scores - mean_vector)
	double mahalanobis_dist = arma::as_scalar(diff.t() * inv_corr_matrix * diff); // Mahalanobis distance
	double term3 = -0.5 * mahalanobis_dist;

	// Total log-likelihood
	return term1 + term2 + term3;
}


// Subset function
arma::vec subset_vector(const arma::vec& ncp_mean_vector, const std::map<int, std::string>& col_idx_SNP) {
	// SNP 인덱스 추출
	arma::uvec indices_to_keep(col_idx_SNP.size());
	size_t idx = 0;
	for (const auto& [key, value] : col_idx_SNP) {
		indices_to_keep(idx++) = key;
	}

	// Subset 벡터 생성
	return ncp_mean_vector(indices_to_keep);
}







double PostCal::fastLikelihood_WC(vector<bool> &a_configure, double NCP) {

	// === [1] ncp mean vector 계산
	arma::vec ncp_mean_vector = get_ncp_mean_vector(ld_matrix.matrix, a_configure);
	ncp_mean_vector.print("NCP mean vector:");

	arma::vec subset_ncp_mean_vector = subset_vector(ncp_mean_vector, ld_matrix.col_idx_SNP);
	subset_ncp_mean_vector.print("Subset NCP mean vector:");


	// === [3] term1, term2 (eigenvalues) 계산.
	double term1 = calc_term1(ld_matrix.M);
	double term2 = calc_term2(ld_matrix.matrix); // 얘가 inverse구하기 전에 계산해야 함.


	// === [4] log-likelihood
	double log_likelihood = calc_log_likelihood_mvn(
		z_observed.z_scores_sorted_vec, subset_ncp_mean_vector, ld_matrix.inv_matrix_subset, term1, term2
	);

	cout << "Log-likelihood: " << log_likelihood << endl;
	printf("%lf\n", log_likelihood);


	return log_likelihood;

}