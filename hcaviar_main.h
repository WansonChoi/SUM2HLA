#ifndef HCAVIAR_MAIN_H
#define HCAVIAR_MAIN_H

#include "hCaviarModel.h"

bool hcaviar_main(
	const std::string& _ldFile, const std::string& _zFile, const std::string& _outputFileName,
	int _totalCausalSNP, double _NCP, double _rho, bool _histFlag, double _gamma,
	int _batch_size=1024
);

// default CAVIAR model
bool hcaviar_main_2(
	const std::string& _ldFile, const std::string& _zFile, const std::string& _outputFileName,
	int _totalCausalSNP, double _NCP, double _rho, bool _histFlag, double _gamma
);

// just fwrite calculated log-likelihood values.
bool hcaviar_main_3(
	const std::string& _ldFile, const std::string& _zFile, const std::string& _outputFileName,
	int _totalCausalSNP, double _NCP, double _rho, bool _histFlag, double _gamma
);

#endif
