CXX = g++
CXXFLAGS_debug = -Wall -g -std=c++17 -DARMA_DONT_USE_WRAPPER
CXXFLAGS = -Wall -O3 -std=c++17 -DARMA_DONT_USE_WRAPPER
#INCLUDES = -I/Users/wansonchoi/Git_Projects/caviar/CAVIAR-C++ -I/Users/wansonchoi/Git_Projects/caviar/CAVIAR-C++/armadillo/include
INCLUDES = -I./ -I./armadillo/include
LDFLAGS = -llapack -lblas -lgslcblas -lgsl

# 소스 파일
SOURCES = main.cpp hcaviar_main.cpp LDMatrix.cpp GWAS_Z_observed.cpp hCaviarModel.cpp Util.cpp

# 객체 파일 디렉토리
OBJ_DIR_RELEASE = release
OBJ_DIR_DEBUG = debug

# 객체 파일 리스트
OBJECTS_RELEASE = $(addprefix $(OBJ_DIR_RELEASE)/, $(SOURCES:.cpp=.o))
OBJECTS_DEBUG = $(addprefix $(OBJ_DIR_DEBUG)/, $(SOURCES:.cpp=.o))

# 타겟
TARGET = hCAVIAR
TARGET_DEBUG = hCAVIAR_debug

# 모듈 1
src_module1 = hcaviar_module1.cpp
obj_module1 = $(src_module1:.cpp=.o)
exec_module1 = hcaviar_module1

# 디렉토리 생성
$(OBJ_DIR_RELEASE):
	mkdir -p $(OBJ_DIR_RELEASE)

$(OBJ_DIR_DEBUG):
	mkdir -p $(OBJ_DIR_DEBUG)

# 최종 타겟 빌드
all: $(TARGET) $(TARGET_DEBUG) $(exec_module1)

# 릴리스 빌드
$(TARGET): $(OBJ_DIR_RELEASE) $(OBJECTS_RELEASE)
	$(CXX) $(CXXFLAGS) $(OBJECTS_RELEASE) $(LDFLAGS) -o $@

# 디버그 빌드
$(TARGET_DEBUG): $(OBJ_DIR_DEBUG) $(OBJECTS_DEBUG)
	$(CXX) $(CXXFLAGS_debug) $(OBJECTS_DEBUG) $(LDFLAGS) -o $@

# 모듈 1 빌드
$(exec_module1): $(obj_module1)
	$(CXX) $(CXXFLAGS) $(obj_module1) $(LDFLAGS) -o $@

# 객체 파일 빌드 규칙 (릴리스)
$(OBJ_DIR_RELEASE)/%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# 객체 파일 빌드 규칙 (디버그)
$(OBJ_DIR_DEBUG)/%.o: %.cpp
	$(CXX) $(CXXFLAGS_debug) $(INCLUDES) -c $< -o $@

# 정리
clean:
	rm -rf $(OBJ_DIR_RELEASE) $(OBJ_DIR_DEBUG) $(obj_module1) $(TARGET) $(TARGET_DEBUG) $(exec_module1)
