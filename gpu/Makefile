CC			= g++
NVCC		= nvcc
CFLAGS		=
TARGET		= gpu
OBJ_FILES	= main.o ctoprim.o hypterm.o diffterm.o helper_functions.o advance.o
HEADERS		= header.h helper_functions.h

$(TARGET):	$(OBJ_FILES)
			$(NVCC) -o $(TARGET) $^

%.o:		%.cpp $(HEADERS)
			$(CC) $(CFLAGS) -c $< -o $@
%.o:		%.cu $(HEADERS)
			$(NVCC) $(CFLAGS) -c $< -o $@
clean:
			rm -f $(OBJ_FILES) $(TARGET)
