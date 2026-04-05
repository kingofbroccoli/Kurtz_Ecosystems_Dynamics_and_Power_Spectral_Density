# Usage:
# make        # compile all binary
# make clean  # remove ALL binaries and objects
# $@ --> target , $< first prerequisite , $^ --> prerequisites

.PHONY: all clean

# Path to sources # Path to Eigen # I could use -I$(EIGEN) instead of having the path in #include
LAPACK_BLAS = -L/home/pietro/University/Programmazione/My_C++/Libraries/ -llapacke -llapack -lrefblas -lgfortran
EIGEN = /home/pietro/University/Programmazione/My_C++/Libraries/eigen-3.4.0/		
TOOLBOX := /home/pietro/University/Programmazione/My_C++/Broccoli_Toolboxes
BUILD_DIR := $(TOOLBOX)/Objects

CC := g++ -std=c++23	# compiler to use
LIBS := -lm $(LAPACK_BLAS)	# external libraries
OPT := -O3 -march=native -mtune=native -fopenmp
CFLAGS := -I$(TOOLBOX) -Wall -Wno-unused-result # -g	# Use -g just for debug			# compiler flags: all warnings + debugger meta-data

# Final binary
BINS := Ecosystem_Dynamics_with_Kurtz_Noise
# Source files
MAIN_SRCS := $(BINS).cpp
# Ecosystem_Dynamics_on_Graphs.cpp
TOOLBOX_SRCS := $(TOOLBOX)/pietro_toolbox.cpp \
				$(TOOLBOX)/matrices_toolbox.cpp \
				$(TOOLBOX)/graph_lv_toolbox.cpp \
				$(TOOLBOX)/graph_from_sequence_toolbox.cpp \
				$(TOOLBOX)/numerical_integration_genLotka-Volterra_toolbox.cpp \
				$(TOOLBOX)/kurtz_bath_eco_toolbox.cpp
SRCS := $(MAIN_SRCS) $(TOOLBOX_SRCS)
# Object files
OBJS := $(addprefix $(BUILD_DIR)/, $(notdir $(SRCS:.cpp=.o)))

all: $(BINS)

# This default rule compiles the executable program
$(BINS): $(OBJS)
	@echo "Checking.."
	$(CC) $(OPT) $(CFLAGS) $^ $(LIBS) -o $@
#	$(CC) $^ $(OPT) $(CFLAGS) $(LIBS) -o $@
	chmod a+x $@

# Compile sources in current directory
$(BUILD_DIR)/%.o: %.cpp
	$(CC) $(OPT) $(CFLAGS) -c $< -o $@

# Compile sources in TOOLBOX directory
$(BUILD_DIR)/%.o: $(TOOLBOX)/%.cpp
	$(CC) $(OPT) $(CFLAGS) -c $< -o $@

clean:
	@echo "Cleaning up..."
	@if [ -d "$(BUILD_DIR)" ]; then rm -vf $(BUILD_DIR)/*.o; fi
	rm -vf $(BINS)