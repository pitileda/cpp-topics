# Name of the project
PROJECT_NAME := hello

# File containing the list of standard module names
STD_MODULES_FILE := std-modules

# File containing the list of module names
MODULES_FILE := modules

# Directory for object files
OBJ_DIR := obj

# Compiler and flags
CXX := g++
CXXFLAGS := -std=c++20 -fmodules-ts

# Read module names from the files
STD_MODULES := $(shell cat $(STD_MODULES_FILE))
MODULES := $(shell cat $(MODULES_FILE))

# Object files for modules
MODULE_OBJ := $(patsubst %.cc,$(OBJ_DIR)/%.o,$(MODULES))

# Main source file
MAIN_SRC := main.cc

# Final executable name
EXECUTABLE := $(PROJECT_NAME)

# Phony targets
.PHONY: all clean

all: $(EXECUTABLE)

# Generate module files for standard modules
gcm:
	@echo "Generating module files for standard modules..."
	@for module in $(STD_MODULES); do \
		$(CXX) $(CXXFLAGS) -xc++-system-header $$module -o /dev/null; \
	done
	@touch gcm

# Compile and generate object files for user-defined modules
$(OBJ_DIR)/%.o: %.cc
	@echo "Generating module files for custom modules..."
	@mkdir -p $(OBJ_DIR)
	@$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile the main source file
$(OBJ_DIR)/$(MAIN_SRC:.cc=.o): $(MAIN_SRC)
	@mkdir -p $(OBJ_DIR)
	@$(CXX) $(CXXFLAGS) -c $< -o $@

# Link the final executable
$(EXECUTABLE): gcm $(MODULE_OBJ) $(OBJ_DIR)/$(MAIN_SRC:.cc=.o)
	@echo "Linking all together..."
	@$(CXX) $(CXXFLAGS) $^ -o $@

clean:
	@rm -rf $(OBJ_DIR) $(EXECUTABLE) gcm.cache gcm

# Rule to build standard modules and generate module files
build_std_modules: $(STD_MODULES_FILE)

# Rule to build user-defined modules
build_modules: $(MODULE_OBJ)
