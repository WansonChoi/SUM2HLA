#include <iostream>
#include <map>
#include <fstream>
#include <iomanip>
#include "Util.h"
#include "hCaviarModel.h"
using namespace std;


unsigned long long comb(int n, int k) {

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


// File export관련.
namespace Util {

	data::data(double num, int ind1, int ind2) {
		number = num;
		index1 = ind1;
		index2 = ind2;
	}

	bool by_number::operator()(data const &left, data const &right) {
		return abs(left.number) > abs(right.number);
	}

}



void printSet2File(const std::string& outputFileName, const vector<char>& pcausalSet, const map<int, string>& snpNames) {

	ofstream outputFile;
	// string outFileNameSet = string(outputFileName)+"_set";

	// outputFile.open(outFileNameSet.c_str());
	outputFile.open(outputFileName.c_str());

	for(int i = 0; i < pcausalSet.size(); i++) {
		if(pcausalSet[i] == '1')
			outputFile << snpNames.at(i) << endl;
	}

}



void printPost2File(const string& outputFileName, const vector<double>& _postValues, const map<int, string>& _snpNames,
	double _total_post, double _totalLikeLihoodLOG) {

	// ofstream outfile(fileName.c_str(), ios::out );
	ofstream outfile(outputFileName.c_str(), ios::out);

	// for(int i = 0; i < pcausalSet.size(); i++)
	// 	total_post = hCaviarModel::addlogSpace(total_post, postValues[i]);

	outfile << "SNP_ID\tProb_in_pCausalSet\tCausal_Post._Prob." << "\n";
	for(int i = 0; i < _postValues.size(); i++) {
		outfile << _snpNames.at(i) << "\t" << exp(_postValues[i] - _total_post) << "\t" << exp(_postValues[i] - _totalLikeLihoodLOG) << "\n";
	}
}


void printHist2File(const string& outputFileName, const vector<double>& _histValues, int _maxCausalSNP,
	double _totalLikeLihoodLOG)
{
	ofstream outfile(outputFileName.c_str(), ios::out | ios::app);

	outfile << "N_causal" << " " << "Post._Prob." << "\n";

	for (int i = 1; i <= _maxCausalSNP; i++)
		outfile << i << " " << exp(_histValues[i] - _totalLikeLihoodLOG) << "\n";

	outfile.close();

}



void printHist2File(const string& outputFileName, const vector<double>& _histValues, int _maxCausalSNP) {
	exportVector2File(outputFileName, _histValues, _maxCausalSNP+1);
}



void export2File(const string& fileName, double data) {
	ofstream outfile(fileName.c_str(), ios::out | ios::app);
	outfile << std::scientific << setprecision(20) << data << endl;
	outfile.close();
}

void export2File(const string& fileName, int data) {
	ofstream outfile(fileName.c_str(), ios::out | ios::app);
	outfile << data << endl;
	outfile.close();
}



string current_time() {
	auto now = std::chrono::system_clock::now();                 // 현재 시스템 시간
	auto now_time = std::chrono::system_clock::to_time_t(now);   // 시간_t 형식으로 변환
	auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
		now.time_since_epoch()) % 1000;                          // 밀리초 추출

	std::ostringstream oss;
	oss << std::put_time(std::localtime(&now_time), "%Y-%m-%d %H:%M:%S") // 형식 지정
		<< "." << std::setfill('0') << std::setw(3) << now_ms.count();   // 밀리초 추가
	return oss.str();
}
