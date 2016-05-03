# Upper directory makefile
SHELL=/bin/bash

default: main.x

main.x:
	$(MAKE) -C src/
	mv src/main.x .

clean:
	$(MAKE) -C src/ clean
	rm -r *.x 
