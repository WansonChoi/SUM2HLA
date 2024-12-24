// hCAVIAR_fwrite

#include <iostream>
#include <string>
#include <unistd.h>
#include <chrono>

# include "hcaviar_main.h"
#include "Util.h"
using namespace std;

int main(int argc, char *argv[]){

	ios::sync_with_stdio(false);

	int totalCausalSNP = 2;
	double NCP = 5.2;
	double gamma = 0.01;
	double rho = 0.95;
	bool histFlag = false;
	int oc = 0;
	string ldFile = "";
	string zFile  = "";
	string outputFileName = "";
	string geneMapFile = "";

	while ((oc = getopt(argc, argv, "vhl:o:z:g:r:c:f:n:")) != -1) {
		switch (oc) {
			case 'v':
				cout << "version 2.2:" << endl;
			case 'h':
				cout << "Options: " << endl;
  				cout << "-h, --help            		show this help message and exit " << endl;
  				cout << "-o OUTFILE, --out=OUTFILE 	specify the output file" << endl;
  				cout << "-l LDFILE, --ld_file=LDFILE  	the ld input file" << endl;
  				cout << "-z ZFILE, --z_file=ZFILE	the z-score and rsID files" << endl;
  				cout << "-r RHO, --rho-prob=RHO		set $pho$ probability (default 0.95)" << endl;
				cout << "-g GAMMA, --gamma		set $gamma$ the prior of a SNP being causal (default 0.01)" << endl;
				cout << "-c causal			set the maximum number of causal SNPs" << endl;
				cout << "-f 1				to out the probaility of different number of causal SNP" << endl;
				cout << "-n NCP				set the Non-Centrality Parameter value" << endl;
				exit(0);
			case 'l':
				ldFile = string(optarg);
				break;
			case 'o':
				outputFileName = string(optarg);
				break;
			case 'z':
				zFile = string(optarg);
				break;
			case 'r':
				rho = atof(optarg);
				break;
			case 'c':
				totalCausalSNP = atoi(optarg);
				break;
			case 'g':
				gamma = atof(optarg);
				break;
			case 'f':
                histFlag = true;
                break;
			case ':':
			case '?':
			case 'n':
				try {
                    NCP = std::stod(optarg);
                } catch (const std::invalid_argument& e) {
                    std::cerr << "Error: Invalid argument for -n. Must be a valid number." << std::endl;
                    exit(1);
                } catch (const std::out_of_range& e) {
                    std::cerr << "Error: Argument for -n is out of range." << std::endl;
                    exit(1);
                }
                break;
			default:
				cout << "Strange" << endl;
				break;
		}
	}


	//program is running
	cout << "@-------------------------------------------------------------@" << endl;
	cout << "| hCAVIAR!		| 	   beta version         |  18/Dec/2024| " << endl;
	cout << "|-------------------------------------------------------------|" << endl;
	cout << "| (C) 2024 Wanson Choi, GNU General Public License, v2 |" << endl;
	cout << "|-------------------------------------------------------------|" << endl;
	cout << "| For documentation, citation & bug-report instructions:      |" << endl;
	cout << "| 		http://genetics.cs.ucla.edu/caviar/           |" << endl;
	cout << "@-------------------------------------------------------------@" << endl;

	// for test (Hard-coding)
	// ldFile = "./sample_data/50_LD.header.PSD.txt";
	// zFile  = "./sample_data/50_Z.header.txt";
	// outputFileName = "./20241218.50_sample.test";

	// ldFile = "./sample_data/C++_example.LD.M10+3.txt";
	// zFile  = "sample_data/C++_example.Z.M10.txt";
	// outputFileName = "./C++_example.20241218.test";

	// ldFile = "./sample_data/IJoined.1kG_vs_Yuki_RA.M23975.Z_fixed.EUR.CLUMP.clumped.M1142+398.PSD.ld";
	// zFile  = "./sample_data/IJoined.1kG_vs_Yuki_RA.M23975.Z_fixed.EUR.CLUMP.CAVIAR_Z";
	// outputFileName = "./sample_data/IJoined.1kG_vs_Yuki_RA.M23975.Z_fixed.EUR.CLUMP.20241218.test";


	cout << "NCP: " << NCP << endl;
    cout << "N_causal: " << totalCausalSNP << " (only)" << endl;

	cout << "\n[" << current_time() << "]: Start.\n";

	hcaviar_main_3(ldFile, zFile, outputFileName,
		totalCausalSNP, NCP, rho, histFlag, gamma);

	cout << "\n[" << current_time() << "]: End.\n";



	return 0;
}
