FROM ubuntu:24.04
LABEL org.opencontainers.image.authors="Tim O'Shea <tim.oshea753@gmail.com>, Peter Bartmann <peter.bartmann@th-luebeck.de>"

# set up environment
ENV DEBIAN_FRONTEND=noninteractive

# update repos/ppas...
RUN apt-get update 
RUN apt-get install -y software-properties-common
RUN apt-add-repository -y ppa:x2go/stable
RUN apt-get update 

# install core packages
RUN apt-get install -y git openssh-server vim emacs screen tmux locate

# Set up remove login info
RUN mkdir /var/run/sshd
RUN echo 'root:radioml' | chpasswd
RUN sed -i 's/PermitRootLogin without-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# somewhat more graphical packages..
RUN apt-get install -y firefox evince audacity meld

# set up remove visual login packages ...
RUN apt-get install -y xfwm4 xfce4 x2goserver x2goserver-xsession

# install python packages
RUN apt-get install -y python3-pip python3-matplotlib python3-scipy python3-numpy python3-dev python3-h5py

# set up gnuradio and related tools
RUN apt-get install -y autotools-dev autoconf sudo wireshark gdb
RUN apt-get install -y gnuradio gnuradio-dev
