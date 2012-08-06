CC			= g++
NVCC		= nvcc

TARGET		= gpu
OBJ_FILES	= main.o ctoprim.o diffterm.o diffterm_cpu.o hypterm.o hypterm_cpu.o hypterm_merged.o helper_functions.o
HEADERS		= header.h helper_functions.h
THRUST		= $(HOME)/software/

BENCHMARK	= advance
B_OBJ_FILES	= benchmark.o advance_cpu.o ctoprim.o diffterm.o diffterm_cpu.o hypterm.o hypterm_cpu.o hypterm_merged.o helper_functions.o

NVCCFLAGS	= -I$(THRUST) -arch=sm_13 --ptxas-options=-v --disable-warnings

$(TARGET):		$(OBJ_FILES)
				$(NVCC) -o $@ $^

$(BENCHMARK):	$(B_OBJ_FILES)
				$(NVCC) -o $@ $^

%.o:		%.cpp $(HEADERS)
			$(CC) $(CFLAGS) -c $< -o $@
%.o:		%.cu $(HEADERS)
			$(NVCC) -c $< -o $@ $(NVCCFLAGS)
clean:
			rm -f $(OBJ_FILES) $(TARGET)
