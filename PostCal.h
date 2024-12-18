#ifndef POSTCAL_H
#define POSTCAL_H

#include <iostream>
#include <fstream>
#include <map>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <armadillo>

#include "LDMatrix.h"
#include "GWAS_Z_observed.h"

using namespace std;
using namespace arma;

void printGSLPrint(mat A, int row, int col);
 
class PostCal{

public:
	double gamma;		// the probability of SNP being causal
	double * postValues;	//the posterior value for each SNP being causal
	double * sigma;		//the LD matrix
	double * histValues;	//the probability of the number of causal SNPs, we make the histogram of the causal SNPs
	int snpCount;		//total number of variants (SNP) in a locus
	int maxCausalSNP;	//maximum number of causal variants to consider in a locus
	double sigmaDet;	//determinie of matrix

	LDMatrix& ld_matrix;
	GWAS_Z_observed& z_observed;

	double totalLikeLihoodLOG; //Compute the total log likelihood of all causal status (by likelihood we use prior)

	// mat sigmaMatrix;
	// mat invSigmaMatrix;
	// mat statMatrix;
 //    mat statMatrixtTran;
	string * snpNames;




	PostCal(LDMatrix& _ld_matrix, GWAS_Z_observed& _z_obs, int snpCount, int maxCausalSNP, string * snpNames, double gamma)
		: ld_matrix(_ld_matrix), z_observed(_z_obs)
	{
		this-> gamma = gamma;
		this-> snpNames = snpNames;
		this-> snpCount = snpCount;
		this-> maxCausalSNP = maxCausalSNP;
        // this-> sigma = new double[snpCount * snpCount];
        this-> sigma = nullptr;
		this-> postValues = new double [snpCount];
		this-> histValues = new double [maxCausalSNP+1];

//		statMatrix                 = mat (snpCount, 1); // M x 1 (z-score vector)
//		statMatrixtTran            = mat (1, snpCount); // 1 x M (z-score vector)
//		sigmaMatrix         	   = mat (snpCount, snpCount); // M x M (LD matrix)
	
//		for(int i = 0; i < snpCount*snpCount; i++)
//			this->sigma[i] = sigma[i];

		for(int i = 0; i < snpCount; i++)
            this->postValues[i] = 0; // 걍 init.이라고 생각하면 될 듯.

		for(int i= 0; i <= maxCausalSNP;i++)
			this->histValues[i] = 0; // 얘도 걍 init.

//		for(int i = 0; i < snpCount; i++) {
//            statMatrix(i,0) = stat[i];
//            statMatrixtTran(0,i) = stat[i];
//	    }
		
//		for(int i = 0; i < snpCount; i++) {
//            for (int j = 0; j < snpCount; j++)
//                sigmaMatrix(i,j) = sigma[i*snpCount+j];
//       	}
		//invSigmaMatrix is depricated and the value for it is not right
		//PLASE DO NOT USE THE invSigmaMatrix;
//		invSigmaMatrix = sigmaMatrix; // 쓰지 말래니까 쓰지 말자.
//		sigmaDet       = det(sigmaMatrix);
	
	}
        ~PostCal() {
		delete [] histValues;
		delete [] postValues;
        delete [] sigma;
	}
 //
	// bool validConfigutation(int * configure, char * pcausalSet);
	// void computeALLCausalSetConfiguration(double * stat, double NCP, char * pcausalSet, string outputFileName);
 //
 //
	// double dmvnorm(mat Z, mat mean, mat R);
 //    double fracdmvnorm(mat Z, mat mean, mat R, mat diagC, double NCP);
	// double fracdmvnorm2(mat Z, mat mean, mat R, mat diagC, double NCP);
 //
 //
 //    double fastLikelihood(int * configure, double * stat, double NCP);
	// double likelihood(int * configure, double * stat, double NCP) ;
	// int nextBinary(int * data, int size) ;
	// double computeTotalLikelihood(double * stat, double NCP) ;
	// double findOptimalSetGreedy(double * stat, double NCP, char * pcausalSet, int *rank,  double inputRho, string outputFileName);
	// string convertConfig2String(int * config, int size);





};
 
#endif
