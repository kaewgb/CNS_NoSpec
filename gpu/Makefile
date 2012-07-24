CC			= g++
NVCC		= nvcc

TARGET		= gpu
OBJ_FILES	= main.o ctoprim.o helper_functions.o
HEADERS		= header.h helper_functions.h
THRUST		= $(HOME)/software/thrust/

NVCCFLAGS	= -I$(THRUST) -arch=sm_13

$(TARGET):	$(OBJ_FILES)
			$(NVCC) -o $(TARGET) $^

%.o:		%.cpp $(HEADERS)
			$(CC) $(CFLAGS) -c $< -o $@
%.o:		%.cu $(HEADERS)
			$(NVCC) -c $< -o $@ $(NVCCFLAGS)
clean:
			rm -f $(OBJ_FILES) $(TARGET)
