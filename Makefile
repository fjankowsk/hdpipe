DCK	    	=   docker

BASEDIR     =   $(CURDIR)
DOCKERFILE  =   ${BASEDIR}/docker/Dockerfile
USERID      =   $(shell id -u)


help:
	@echo 'Makefile for heimdall pipeline'
	@echo 'Usage:'
	@echo 'make interactive     run an interactive shell'
	@echo 'make production      build docker image for production use'

interactive:
	${DCK} run -it --rm \
	--runtime=nvidia \
	--user ${USERID} \
	hdpipe bash

production:
	${DCK} build \
	--build-arg USERID=${USERID} \
	--file ${DOCKERFILE} \
	--tag hdpipe ${BASEDIR}

.PHONY: help production