CC			= g++
TARGET		= advance
OBJ_FILES	= main.o ctoprim.o hypterm.o diffterm.o

$(TARGET):	$(OBJ_FILES)
			$(CC) -o $(TARGET) $^

%.o:		%.cpp header.h
			$(CC) $(CFLAGS) -c $< -o $@
clean:
			rm -f $(OBJ_FILES) $(TARGET)
