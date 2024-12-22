# 컴파일러 설정
CXX = g++
CXXFLAGS_debug = -Wall -g -std=c++17 -DARMA_DONT_USE_WRAPPER
CXXFLAGS = -Wall -O3 -std=c++17 -DARMA_DONT_USE_WRAPPER
INCLUDES = -I./ -I./armadillo/include
LDFLAGS = -llapack -lblas -lgslcblas -lgsl

# 객체 파일 디렉토리 (중앙 관리)
OBJ_BASE_DIR = ./obj
OBJ_DIR_RELEASE = $(OBJ_BASE_DIR)/release
OBJ_DIR_DEBUG = $(OBJ_BASE_DIR)/debug

# 모듈별 객체 파일 디렉토리
MODULE1_DIR = $(OBJ_BASE_DIR)/$(exec_module1)
MODULE2_DIR = $(OBJ_BASE_DIR)/$(exec_module2)

# 소스 파일
SOURCES = main.cpp hcaviar_main.cpp LDMatrix.cpp GWAS_Z_observed.cpp hCaviarModel.cpp Util.cpp

# 객체 파일 리스트
OBJECTS_RELEASE = $(addprefix $(OBJ_DIR_RELEASE)/, $(SOURCES:.cpp=.o))
OBJECTS_DEBUG = $(addprefix $(OBJ_DIR_DEBUG)/, $(SOURCES:.cpp=.o))

# 타겟
TARGET = hCAVIAR
TARGET_DEBUG = hCAVIAR_debug

# 모듈 소스
src_module1 = hcaviar_module1.cpp
src_module2 = hcaviar_module2.cpp LDMatrix.cpp

# 모듈 객체 파일 리스트
obj_module1 = $(addprefix $(MODULE1_DIR)/, $(src_module1:.cpp=.o))
obj_module2 = $(addprefix $(MODULE2_DIR)/, $(src_module2:.cpp=.o))

# 모듈 타겟
exec_module1 = hcaviar_module1
exec_module2 = hcaviar_module2

# 디렉토리 생성 규칙
$(OBJ_DIR_RELEASE) $(OBJ_DIR_DEBUG) $(MODULE1_DIR) $(MODULE2_DIR):
	mkdir -p $@

# 최종 빌드 타겟
all: $(TARGET) $(TARGET_DEBUG) $(exec_module1) $(exec_module2)

# 릴리스 빌드
$(TARGET): $(OBJ_DIR_RELEASE) $(OBJECTS_RELEASE)
	$(CXX) $(CXXFLAGS) $(OBJECTS_RELEASE) $(LDFLAGS) -o $@

# 디버그 빌드
$(TARGET_DEBUG): $(OBJ_DIR_DEBUG) $(OBJECTS_DEBUG)
	$(CXX) $(CXXFLAGS_debug) $(OBJECTS_DEBUG) $(LDFLAGS) -o $@

# 모듈 1 빌드
$(exec_module1): $(MODULE1_DIR) $(obj_module1)
	$(CXX) $(CXXFLAGS_debug) $(obj_module1) $(LDFLAGS) -o $@

# 모듈 2 빌드
$(exec_module2): $(MODULE2_DIR) $(obj_module2)
	$(CXX) $(CXXFLAGS_debug) $(obj_module2) $(LDFLAGS) -o $@

# 객체 파일 빌드 규칙 (릴리스)
$(OBJ_DIR_RELEASE)/%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# 객체 파일 빌드 규칙 (디버그)
$(OBJ_DIR_DEBUG)/%.o: %.cpp
	$(CXX) $(CXXFLAGS_debug) $(INCLUDES) -c $< -o $@

# 객체 파일 빌드 규칙 (모듈 1)
$(MODULE1_DIR)/%.o: %.cpp
	$(CXX) $(CXXFLAGS_debug) $(INCLUDES) -c $< -o $@

# 객체 파일 빌드 규칙 (모듈 2)
$(MODULE2_DIR)/%.o: %.cpp
	$(CXX) $(CXXFLAGS_debug) $(INCLUDES) -c $< -o $@

# 정리
clean:
	rm -rf $(OBJ_BASE_DIR) $(TARGET) $(TARGET_DEBUG) $(exec_module1) $(exec_module2)
