#ifndef UTIL_H
#define UTIL_H


unsigned long long comb(int n, int k);


// File export관련.
namespace Util {

	struct data {
		data(double num, int ind1, int ind2);
		double number;
		int index1;
		int index2;
	};

	struct by_number {
		bool operator()(data const &left, data const &right);
	};

}



void printSet2File(const std::string& outputFileName, const std::vector<char>& _pcausalSet, const std::map<int, std::string>& _snpNames); // added by Wanson Choi.

void printPost2File(const std::string& outputFileName, const std::vector<double>& _postValues, const std::map<int, std::string>& _snpNames,
	double _total_post, double _totalLikeLihoodLOG);

void printHist2File(const std::string& outputFileName, const std::vector<double>& _histValues, int _maxCausalSNP,
	double _totalLikeLihoodLOG);


void printHist2File(const std::string& outputFileName, const std::vector<double>& _histValues, int _maxCausalSNP);

	template <typename T>
	void exportVector2File(const std::string& fileName, const T& data, int size) {
		std::ofstream outfile(fileName.c_str(), std::ios::out | std::ios::app);

		for (int i = 1; i < size; i++)
			outfile << data[i] << " ";
		//outfile << endl;
		outfile.close();
	}

	// void exportVector2File(const std::string& fileName, char * data, int size);
	//
	// void exportVector2File(const std::string& fileName, double * data, int size);
	//
	// void exportVector2File(const std::string& fileName, int * data, int size);


void export2File(const std::string& fileName, double data);
void export2File(const std::string& fileName, int data);


std::string current_time();

#endif //UTIL_H
